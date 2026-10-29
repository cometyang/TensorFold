#include "traffic_profile.h"
#include <cuda.h>
#include <cupti.h>
#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <cupti_profiler_host.h>
#include <cupti_range_profiler.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
#define PARAM(type, name) type name{}; name.structSize = type##_STRUCT_SIZE
void check(CUptiResult r, const char *op) {
    if (r == CUPTI_SUCCESS) return;
    const char *s = nullptr; cuptiGetResultString(r, &s);
    throw std::runtime_error(std::string(op) + ": " + (s ? s : "CUPTI error"));
}
#define CUPTI(call) check((call), #call)
void driver(CUresult r, const char *op) {
    if (r == CUDA_SUCCESS) return;
    const char *s = nullptr; cuGetErrorString(r, &s);
    throw std::runtime_error(std::string(op) + ": " + (s ? s : "CUDA error"));
}
std::string env(const char *key, const char *fallback = "") {
    const char *v = std::getenv(key); return v ? v : fallback;
}
uint64_t number(const char *key, uint64_t fallback) {
    auto s = env(key); if (s.empty()) return fallback;
    size_t n = 0; auto v = std::stoull(s, &n);
    if (n != s.size() || s.front() == '-') throw std::runtime_error(std::string("Invalid ") + key);
    return v;
}
std::vector<std::string> split(const std::string &s) {
    std::vector<std::string> result; std::stringstream in(s); std::string token;
    while (std::getline(in, token, ',')) {
        auto a = token.find_first_not_of(" \t\r\n"), b = token.find_last_not_of(" \t\r\n");
        if (a != std::string::npos) result.push_back(token.substr(a, b-a+1));
    }
    return result;
}
std::string json(const std::string &s) {
    std::ostringstream o; o << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') o << '\\' << c;
        else if (c < 32) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
        else o << c;
    }
    o << '"'; return o.str();
}
struct Host {
    CUpti_Profiler_Host_Object *object = nullptr;
    CUcontext context = nullptr;
    std::string chip;
    std::vector<uint8_t> availability, config;
    void init(const char *offline_chip = nullptr) {
        if (offline_chip) { chip = offline_chip; }
        else {
        driver(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
        if (!context) throw std::runtime_error("No current CUDA context");
        CUdevice device; driver(cuCtxGetDevice(&device), "cuCtxGetDevice");
        PARAM(CUpti_Profiler_Initialize_Params, init); CUPTI(cuptiProfilerInitialize(&init));
        PARAM(CUpti_Device_GetChipName_Params, c); c.deviceIndex = device;
        CUPTI(cuptiDeviceGetChipName(&c)); chip = c.pChipName;
        PARAM(CUpti_Profiler_GetCounterAvailability_Params, a); a.ctx = context;
        a.bAllowDeviceLevelCounters = 1;
        CUPTI(cuptiProfilerGetCounterAvailability(&a)); availability.resize(a.counterAvailabilityImageSize);
        a.pCounterAvailabilityImage = availability.data(); CUPTI(cuptiProfilerGetCounterAvailability(&a));
        }
        PARAM(CUpti_Profiler_Host_Initialize_Params, h);
        h.profilerType = CUPTI_PROFILER_TYPE_RANGE_PROFILER;
        h.pChipName = chip.c_str(); h.pCounterAvailabilityImage = availability.empty() ? nullptr : availability.data();
        CUPTI(cuptiProfilerHostInitialize(&h)); object = h.pHostObject;
    }
    void release() noexcept {
        if (!object) return;
        PARAM(CUpti_Profiler_Host_Deinitialize_Params, d); d.pHostObject = object;
        cuptiProfilerHostDeinitialize(&d); object = nullptr;
    }
    CUpti_Profiler_Host_GetMetricProperties_Params properties(const char *name) {
        PARAM(CUpti_Profiler_Host_GetMetricProperties_Params, p);
        p.pHostObject = object; p.pMetricName = name; CUPTI(cuptiProfilerHostGetMetricProperties(&p)); return p;
    }
    size_t configure(std::vector<const char*> &names) {
        PARAM(CUpti_Profiler_Host_ConfigAddMetrics_Params, a);
        a.pHostObject = object; a.ppMetricNames = names.data(); a.numMetrics = names.size();
        CUPTI(cuptiProfilerHostConfigAddMetrics(&a));
        PARAM(CUpti_Profiler_Host_GetConfigImageSize_Params, s); s.pHostObject = object;
        CUPTI(cuptiProfilerHostGetConfigImageSize(&s)); config.resize(s.configImageSize);
        PARAM(CUpti_Profiler_Host_GetConfigImage_Params, g);
        g.pHostObject = object; g.configImageSize = config.size(); g.pConfigImage = config.data();
        CUPTI(cuptiProfilerHostGetConfigImage(&g));
        PARAM(CUpti_Profiler_Host_GetNumOfPasses_Params, p);
        p.pConfigImage = config.data(); p.configImageSize = config.size();
        CUPTI(cuptiProfilerHostGetNumOfPasses(&p)); return p.numOfPasses;
    }
};
std::string properties_json(Host &host, const char *name) {
    auto p = host.properties(name);
    std::ostringstream o; o << "{\"name\":" << json(name) << ",\"description\":" << json(p.pDescription ? p.pDescription : "")
        << ",\"unit\":" << json(p.pDimUnit ? p.pDimUnit : "") << ",\"hw_unit\":" << json(p.pHwUnit ? p.pHwUnit : "")
        << ",\"scope\":" << int(p.metricCollectionScope) << ",\"type\":" << int(p.metricType) << '}';
    return o.str();
}
struct Round { std::string name; uint64_t context; uint32_t rows; int rank; };
struct State {
    Host host;
    std::mutex mutex;
    bool configured = false, failed = false, enabled = false, started = false, pushed = false;
    int rank = -1, bucket = -1;
    std::thread::id owner;
    uint64_t limit = 8, skip = 2, minimum = 0, maximum = UINT64_MAX, max_rows = 16;
    std::array<uint64_t, 3> seen{}, collected{};
    std::vector<std::string> metrics;
    std::vector<const char*> names;
    std::vector<uint8_t> data;
    std::vector<Round> rounds;
    std::ofstream out;
    CUpti_RangeProfiler_Object *profiler = nullptr;
    void emit(const std::string &line) { if (out.is_open()) { out << line << '\n'; out.flush(); } }
    void configure(int r) {
        rank = r;
        std::string path = env("TF_TRAFFIC_OUTPUT"), supplied = env("TF_TRAFFIC_METRICS");
        if (path.empty() || supplied.empty()) throw std::runtime_error("TF_TRAFFIC_OUTPUT and TF_TRAFFIC_METRICS required");
        auto marker = path.find("%r");
        if (marker == std::string::npos) path += ".rank" + std::to_string(rank) + ".jsonl";
        else path.replace(marker, 2, std::to_string(rank));
        out.open(path, std::ios::app); if (!out) throw std::runtime_error("Cannot open traffic output");
        limit = number("TF_TRAFFIC_RANGES", 8); skip = number("TF_TRAFFIC_SKIP", 2);
        minimum = number("TF_TRAFFIC_MIN_CONTEXT", 0); maximum = number("TF_TRAFFIC_MAX_CONTEXT", UINT64_MAX);
        max_rows = number("TF_TRAFFIC_MAX_ROWS", 16);
        if (!limit || limit > 1024 || minimum > maximum || !max_rows) throw std::runtime_error("Invalid traffic limits");
        metrics = split(supplied); if (metrics.empty() || metrics.size() > 64) throw std::runtime_error("Invalid metric count");
        for (auto &m : metrics) names.push_back(m.c_str());
        host.init();
        for (auto name : names) emit("{\"type\":\"metric_metadata\",\"metric\":" + properties_json(host,name) + "}");
        size_t passes = host.configure(names);
        emit("{\"type\":\"configuration\",\"rank\":" + std::to_string(rank) + ",\"chip\":" + json(host.chip) +
            ",\"passes\":" + std::to_string(passes) + ",\"run\":" + json(env("TF_TRAFFIC_RUN")) +
            ",\"bucket_label\":" + json(env("TF_TRAFFIC_BUCKET")) + ",\"mode\":\"UserRange/UserReplay; no replay\"}");
        if (passes != 1) throw std::runtime_error("Metric configuration is not single-pass; no profiling started");
        configured = true;
    }
    void start(int b) {
        PARAM(CUpti_RangeProfiler_Enable_Params, e); e.ctx = host.context;
        CUPTI(cuptiRangeProfilerEnable(&e)); profiler = e.pRangeProfilerObject; enabled = true;
        PARAM(CUpti_RangeProfiler_GetCounterDataSize_Params, s);
        s.pRangeProfilerObject = profiler; s.pMetricNames = names.data(); s.numMetrics = names.size();
        s.maxNumOfRanges = limit; s.maxNumRangeTreeNodes = limit;
        CUPTI(cuptiRangeProfilerGetCounterDataSize(&s)); data.assign(s.counterDataSize,0);
        PARAM(CUpti_RangeProfiler_CounterDataImage_Initialize_Params, i);
        i.pRangeProfilerObject = profiler; i.pCounterData = data.data(); i.counterDataSize = data.size();
        CUPTI(cuptiRangeProfilerCounterDataImageInitialize(&i));
        PARAM(CUpti_RangeProfiler_SetConfig_Params, c);
        c.pRangeProfilerObject = profiler; c.pConfig = host.config.data(); c.configSize = host.config.size();
        c.pCounterDataImage = data.data(); c.counterDataImageSize = data.size();
        c.range = CUPTI_UserRange; c.replayMode = CUPTI_UserReplay; c.maxRangesPerPass = limit;
        c.numNestingLevels = 1; c.minNestingLevel = 1; c.passIndex = 0; c.targetNestingLevel = 1;
        CUPTI(cuptiRangeProfilerSetConfig(&c));
        PARAM(CUpti_RangeProfiler_Start_Params, p); p.pRangeProfilerObject = profiler;
        CUPTI(cuptiRangeProfilerStart(&p)); started = true; bucket = b; rounds.clear();
    }
    void finish_window() {
        if (!started) return;
        if (pushed) throw std::runtime_error("finish called inside an open range");
        PARAM(CUpti_RangeProfiler_Stop_Params, s); s.pRangeProfilerObject = profiler;
        CUPTI(cuptiRangeProfilerStop(&s)); started = false;
        if (s.isAllPassSubmitted != 1) throw std::runtime_error("Incomplete pass; measurement discarded, no replay");
        PARAM(CUpti_RangeProfiler_DecodeData_Params, d); d.pRangeProfilerObject = profiler;
        CUPTI(cuptiRangeProfilerDecodeData(&d));
        if (d.numOfRangeDropped) throw std::runtime_error("Dropped ranges; measurement discarded");
        PARAM(CUpti_RangeProfiler_GetCounterDataInfo_Params, info);
        info.pCounterDataImage = data.data(); info.counterDataImageSize = data.size();
        CUPTI(cuptiRangeProfilerGetCounterDataInfo(&info));
        if (info.numTotalRanges != rounds.size()) throw std::runtime_error("Range count mismatch; measurement discarded");
        std::vector<std::string> receipts;
        for (size_t n = 0; n < rounds.size(); ++n) {
            PARAM(CUpti_RangeProfiler_CounterData_GetRangeInfo_Params, ri);
            ri.pCounterDataImage = data.data(); ri.counterDataImageSize = data.size(); ri.rangeIndex = n; ri.rangeDelimiter = "/";
            CUPTI(cuptiRangeProfilerCounterDataGetRangeInfo(&ri));
            auto r = std::find_if(rounds.begin(),rounds.end(),[&](const Round &v){return v.name == ri.rangeName;});
            if (r == rounds.end()) throw std::runtime_error("Range identity mismatch");
            std::vector<double> values(names.size());
            PARAM(CUpti_Profiler_Host_EvaluateToGpuValues_Params, v);
            v.pHostObject = host.object; v.pCounterDataImage = data.data(); v.counterDataImageSize = data.size();
            v.rangeIndex = n; v.ppMetricNames = names.data(); v.numMetrics = names.size(); v.pMetricValues = values.data();
            CUPTI(cuptiProfilerHostEvaluateToGpuValues(&v));
            std::ostringstream line; line << std::setprecision(17) << "{\"type\":\"traffic_round\",\"range\":" << json(r->name)
                << ",\"context_tokens\":" << r->context << ",\"rows\":" << r->rows << ",\"rank\":" << r->rank << ",\"values\":{";
            for (size_t j = 0; j < names.size(); ++j) {
                if (!std::isfinite(values[j])) throw std::runtime_error("Nonfinite metric; window discarded");
                if (j) line << ','; line << json(names[j]) << ':' << values[j];
            }
            line << "}}"; receipts.push_back(line.str());
        }
        PARAM(CUpti_RangeProfiler_Disable_Params, disable); disable.pRangeProfilerObject = profiler;
        CUPTI(cuptiRangeProfilerDisable(&disable)); enabled = false; profiler = nullptr;
        for (auto &line : receipts) emit(line);
        emit("{\"type\":\"window_complete\",\"bucket\":" + std::to_string(bucket) + ",\"ranges\":" + std::to_string(rounds.size()) + "}");
        rounds.clear(); bucket = -1;
    }
    void fail(const char *reason) noexcept {
        failed = true;
        std::fprintf(stderr,"TensorFold traffic profiling disabled: %s\n",reason);
        try { emit("{\"type\":\"error\",\"message\":" + json(reason) + "}"); } catch (...) {}
        // Best-effort cleanup never submits workload or attempts replay.
        if (pushed) { PARAM(CUpti_RangeProfiler_PopRange_Params,p); p.pRangeProfilerObject=profiler; cuptiRangeProfilerPopRange(&p); pushed=false; }
        if (started) { PARAM(CUpti_RangeProfiler_Stop_Params,s); s.pRangeProfilerObject=profiler; cuptiRangeProfilerStop(&s); started=false; }
        if (enabled) { PARAM(CUpti_RangeProfiler_Disable_Params,d); d.pRangeProfilerObject=profiler; cuptiRangeProfilerDisable(&d); enabled=false; profiler=nullptr; }
        host.release();
    }
};
// No process-exit GPU callbacks: finish must run while the context is alive.
State &state() { static State *s = new State; return *s; }
}

extern "C" int tf_traffic_begin(uint64_t tokens, uint32_t rows, int rank) {
    if (env("TF_TRAFFIC_ENABLE") != "1") return 0;
    auto &s=state(); std::lock_guard<std::mutex> lock(s.mutex);
    if (s.failed) return -1;
    try {
        if (!s.configured) s.configure(rank);
        if (s.pushed) throw std::runtime_error("Overlapping traffic ranges");
        CUcontext context; driver(cuCtxGetCurrent(&context),"cuCtxGetCurrent");
        if (context != s.host.context || rank != s.rank) throw std::runtime_error("CUDA context/rank changed");
        if (!rows || rows>s.max_rows || tokens<s.minimum || tokens>s.maximum) return 0;
        int b = tokens<=4096 ? 0 : tokens<=131072 ? 1 : 2;
        if (s.started && s.bucket != b) s.finish_window();
        if (s.seen[b]++ < s.skip || s.collected[b]>=s.limit) return 0;
        if (!s.started) s.start(b);
        std::string name="rank"+std::to_string(rank)+"_b"+std::to_string(b)+"_round"+std::to_string(s.seen[b]-1);
        PARAM(CUpti_RangeProfiler_PushRange_Params,p); p.pRangeProfilerObject=s.profiler; p.pRangeName=name.c_str();
        CUPTI(cuptiRangeProfilerPushRange(&p)); s.pushed=true; s.owner=std::this_thread::get_id();
        s.rounds.push_back({name,tokens,rows,rank}); return 1;
    } catch (const std::exception &e) { s.fail(e.what()); return -1; }
}
extern "C" int tf_traffic_end() {
    if (env("TF_TRAFFIC_ENABLE") != "1") return 0;
    auto &s=state(); std::lock_guard<std::mutex> lock(s.mutex);
    if (s.failed) return -1; if (!s.pushed) return 0;
    try {
        if (s.owner!=std::this_thread::get_id()) throw std::runtime_error("Range ended on another host thread");
        PARAM(CUpti_RangeProfiler_PopRange_Params,p); p.pRangeProfilerObject=s.profiler;
        CUPTI(cuptiRangeProfilerPopRange(&p)); s.pushed=false; ++s.collected[s.bucket];
        if (s.collected[s.bucket]>=s.limit) s.finish_window();
        return 1;
    } catch (const std::exception &e) { s.fail(e.what()); return -1; }
}
extern "C" void tf_traffic_finish() {
    if (env("TF_TRAFFIC_ENABLE") != "1") return;
    auto &s=state(); std::lock_guard<std::mutex> lock(s.mutex);
    if (s.failed) return;
    try { s.finish_window(); } catch (const std::exception &e) { s.fail(e.what()); }
}
static int query(const char *filters, const char *chip) {
    Host h;
    try {
        h.init(chip); auto terms=split(filters ? filters : "");
        std::printf("{\"type\":\"chip\",\"name\":%s}\n",json(h.chip).c_str());
        for (int type=0;type<int(CUPTI_METRIC_TYPE__COUNT);++type) {
            PARAM(CUpti_Profiler_Host_GetBaseMetrics_Params,b); b.pHostObject=h.object; b.metricType=CUpti_MetricType(type);
            CUPTI(cuptiProfilerHostGetBaseMetrics(&b));
            for (size_t n=0;n<b.numMetrics;++n) {
                std::string name=b.ppMetricNames[n];
                if (!terms.empty() && std::none_of(terms.begin(),terms.end(),[&](const std::string &t){return name.find(t)!=std::string::npos;})) continue;
                auto line=properties_json(h,name.c_str());
                PARAM(CUpti_Profiler_Host_GetSubMetrics_Params,sub); sub.pHostObject=h.object; sub.metricType=CUpti_MetricType(type); sub.pMetricName=name.c_str();
                CUPTI(cuptiProfilerHostGetSubMetrics(&sub)); line.pop_back(); line+=",\"suffixes\":[";
                for (size_t k=0;k<sub.numOfSubmetrics;++k) { if(k)line+=","; line+=json(sub.ppSubMetrics[k]); }
                line+="]}"; std::puts(line.c_str());
            }
        }
        // Optional requested configuration validates pass count without enabling profiling.
        auto metrics=split(env("TF_TRAFFIC_METRICS")); std::vector<const char*> names;
        for(auto &m:metrics) names.push_back(m.c_str());
        if(!names.empty()) std::printf("{\"type\":\"passes\",\"count\":%zu}\n",h.configure(names));
        h.release(); return 0;
    } catch(const std::exception &e) { std::fprintf(stderr,"traffic query: %s\n",e.what()); h.release(); return 1; }
}
extern "C" int tf_traffic_query(const char *filters) { return query(filters,nullptr); }
extern "C" int tf_traffic_query_chip(const char *chip, const char *filters) { return query(filters,chip); }
