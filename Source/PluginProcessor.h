/*
  ==============================================================================
    PhaseCorrector - Freeform Phase EQ Plugin
    Low-THD FFT-based phase manipulation with 4x Overlap-Add architecture
    Inspired by MFreeformPhase with superior audio quality
  ==============================================================================
*/

#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <array>
#include <vector>
#include <complex>
#include <atomic>
#include <mutex>

//==============================================================================
// Cubic Spline Interpolation for smooth phase curves
//==============================================================================
class CubicSpline
{
public:
    CubicSpline() = default;

    bool build(const std::vector<double>& x, const std::vector<double>& y);
    double evaluate(double x) const;
    bool isValid() const { return valid && !xData.empty(); }
    void clear();

private:
    std::vector<double> xData, yData;
    std::vector<double> a, b, c, d;
    bool valid = false;
};

//==============================================================================
// Nyquist Filter - Cascaded Biquad Low-Pass
//==============================================================================
class NyquistFilter
{
public:
    static constexpr int MAX_STAGES = 8; // Up to 96 dB/octave

    NyquistFilter();

    void prepare(double sampleRate);
    void reset();
    void setParameters(float frequency, float q, int stages);
    float processSample(float sample, int channel);

    // For visualization
    float getMagnitudeAtFrequency(double freq) const;
    float getFrequency() const { return cutoffFreq; }
    float getQ() const { return resonance; }
    int getStages() const { return numStages; }

private:
    void updateCoefficients();

    struct BiquadState
    {
        double x1 = 0, x2 = 0;
        double y1 = 0, y2 = 0;
    };

    std::array<std::array<BiquadState, MAX_STAGES>, 2> states; // Per channel, per stage

    // Coefficients (same for all stages)
    double b0 = 1, b1 = 0, b2 = 0;
    double a1 = 0, a2 = 0;

    double currentSampleRate = 44100.0;
    float cutoffFreq = 20000.0f;
    float resonance = 0.707f;
    int numStages = 2;
};

//==============================================================================
// Fast Math Approximations for Real-time Audio
//==============================================================================
namespace FastMath
{
    // Fast inverse square root (Quake III style, improved)
    inline float fastInvSqrt(float x)
    {
        float xhalf = 0.5f * x;
        int i = *(int*)&x;
        i = 0x5f375a86 - (i >> 1);
        x = *(float*)&i;
        x = x * (1.5f - xhalf * x * x);  // One Newton-Raphson iteration
        return x;
    }

    // Fast square root using fast inverse sqrt
    inline float fastSqrt(float x)
    {
        if (x <= 0.0f) return 0.0f;
        return x * fastInvSqrt(x);
    }

    // Fast atan2 approximation (max error ~0.01 radians)
    inline float fastAtan2(float y, float x)
    {
        const float PI = 3.14159265358979f;
        const float PI_2 = 1.57079632679489f;

        if (x == 0.0f)
        {
            if (y > 0.0f) return PI_2;
            if (y < 0.0f) return -PI_2;
            return 0.0f;
        }

        float abs_y = std::abs(y) + 1e-10f;  // Prevent division by zero
        float angle;

        if (x >= 0.0f)
        {
            float r = (x - abs_y) / (x + abs_y);
            angle = 0.1963f * r * r * r - 0.9817f * r + PI / 4.0f;
        }
        else
        {
            float r = (x + abs_y) / (abs_y - x);
            angle = 0.1963f * r * r * r - 0.9817f * r + 3.0f * PI / 4.0f;
        }

        return (y < 0.0f) ? -angle : angle;
    }

    // Fast sin approximation using parabola (max error ~0.001)
    inline float fastSin(float x)
    {
        const float PI = 3.14159265358979f;
        const float TWO_PI = 6.28318530717959f;

        // Normalize to [-PI, PI]
        x = std::fmod(x + PI, TWO_PI) - PI;

        const float B = 4.0f / PI;
        const float C = -4.0f / (PI * PI);
        const float P = 0.225f;

        float y = B * x + C * x * std::abs(x);
        return P * (y * std::abs(y) - y) + y;
    }

    // Fast cos using sin
    inline float fastCos(float x)
    {
        return fastSin(x + 1.57079632679489f);
    }
}

//==============================================================================
// FFT-based Phase Processor with Overlap-Add
//==============================================================================
class PhaseProcessor
{
public:
    // Quality levels - determines FFT size and frequency resolution
    enum class Quality
    {
        Low = 0,      // 2048 FFT, ~43 Hz resolution @ 44.1k
        Medium = 1,   // 4096 FFT, ~21 Hz resolution
        High = 2,     // 8192 FFT, ~10.7 Hz resolution (default)
        VeryHigh = 3, // 16384 FFT, ~5.4 Hz resolution
        Extreme = 4   // 32768 FFT, ~2.7 Hz resolution
    };

    // Overlap amount - affects latency and quality
    enum class Overlap
    {
        Percent50 = 0,  // 2x overlap, lower latency, uses Sine window
        Percent75 = 1   // 4x overlap, better quality, uses Hann window (default)
    };

    PhaseProcessor();
    ~PhaseProcessor() = default;

    void prepare(double sampleRate, int maxBlockSize);
    void process(juce::AudioBuffer<float>& buffer);
    void reset();

    void updatePhaseCurve(const std::vector<std::pair<double, double>>& points);
    void setDepth(float depth) { phaseDepth.store(juce::jlimit(-2.0f, 2.0f, depth)); }
    float getDepth() const { return phaseDepth.load(); }

    // Quality and overlap settings
    void setQuality(Quality q);
    void setOverlap(Overlap o);
    Quality getQuality() const { return currentQuality; }
    Overlap getOverlap() const { return currentOverlap; }

    // Get current latency in samples
    int getLatencySamples() const { return analysisSize; }

    const std::vector<float>& getPhaseTable() const { return phaseTable; }
    double getSampleRate() const { return currentSampleRate; }
    int getFFTSize() const { return fftSize; }
    int getAnalysisSize() const { return analysisSize; }

    static juce::StringArray getQualityNames();
    static juce::StringArray getOverlapNames();

private:
    void rebuildPhaseTable();
    void processFrame(int channel);
    void processFrameBypass(int channel);  // Fast path when no phase processing needed
    void reconfigure();  // Rebuilds FFT, windows, buffers
    void buildWindows();

    std::unique_ptr<juce::dsp::FFT> fft;

    struct ChannelState
    {
        std::vector<float> inputBuffer;
        std::vector<float> outputBuffer;
        std::vector<float> fftBuffer;  // Pre-allocated FFT workspace
        int inputWritePos = 0;
        int outputReadPos = 0;
        int samplesUntilNextFrame = 0;

        void resize(int bufferSize, int fftSize, int hopSize)
        {
            inputBuffer.resize(bufferSize, 0.0f);
            outputBuffer.resize(bufferSize, 0.0f);
            fftBuffer.resize(fftSize * 2, 0.0f);  // Real + imaginary
            inputWritePos = 0;
            outputReadPos = 0;
            samplesUntilNextFrame = hopSize;
        }

        void clear()
        {
            std::fill(inputBuffer.begin(), inputBuffer.end(), 0.0f);
            std::fill(outputBuffer.begin(), outputBuffer.end(), 0.0f);
            inputWritePos = 0;
            outputReadPos = 0;
        }
    };

    std::array<ChannelState, 2> channels;

    // Dynamic windows
    std::vector<float> analysisWindow;
    std::vector<float> synthesisWindow;
    float windowCompensation = 1.0f;

    // Current FFT parameters
    int fftOrder = 13;        // 2^13 = 8192
    int fftSize = 8192;
    int analysisSize = 4096;
    int hopSize = 1024;
    int numOverlaps = 4;

    // Quality settings
    Quality currentQuality = Quality::High;
    Overlap currentOverlap = Overlap::Percent75;
    std::atomic<bool> needsReconfigure{false};
    juce::SpinLock processingLock;

    // Phase modification table
    std::vector<float> phaseTable;
    CubicSpline phaseCurve;
    std::mutex curveMutex;

    double currentSampleRate = 44100.0;
    std::atomic<float> phaseDepth{1.0f};

    static constexpr float MIN_FREQ = 20.0f;
    static constexpr float MAX_FREQ = 20000.0f;
    static constexpr float LOG_MIN_FREQ = 1.301030f;
    static constexpr float LOG_MAX_FREQ = 4.301030f;
};

//==============================================================================
// CSV Parser for phase curves
//==============================================================================
class CSVParser
{
public:
    struct ParseResult
    {
        bool success = false;
        std::vector<std::pair<double, double>> points;
        juce::String errorMessage;
        int pointCount = 0;
    };

    static ParseResult parse(const juce::File& file);
    static ParseResult parseString(const juce::String& content);

private:
    static constexpr float LOG_MIN_FREQ = 1.301030f;
    static constexpr float LOG_MAX_FREQ = 4.301030f;
};

//==============================================================================
// Oversampling Manager
//==============================================================================
class OversamplingManager
{
public:
    enum class Rate
    {
        x1 = 0,
        x2 = 1,
        x4 = 2,
        x8 = 3,
        x16 = 4,
        x32 = 5,
        x64 = 6
    };

    enum class FilterMode
    {
        FIR = 0,  // Linear phase
        IIR = 1   // Minimum phase (lower CPU)
    };

    OversamplingManager();
    ~OversamplingManager() = default;

    void prepare(double sampleRate, int blockSize);
    void reset();

    // Call this at the start of processBlock to apply any pending changes
    // Returns true if rate/mode changed (dependents need update)
    bool applyPendingChanges();

    juce::dsp::AudioBlock<float> processSamplesUp(juce::dsp::AudioBlock<float>& inputBlock);
    void processSamplesDown(juce::dsp::AudioBlock<float>& outputBlock);

    void setRate(Rate newRate);
    void setFilterMode(FilterMode mode);
    Rate getRate() const { return currentRate; }
    FilterMode getFilterMode() const { return filterMode; }
    int getFactor() const { return 1 << static_cast<int>(currentRate); }
    float getLatencySamples() const;

    static juce::StringArray getRateNames();
    static juce::StringArray getFilterModeNames();

private:
    void createOversampler();

    std::unique_ptr<juce::dsp::Oversampling<float>> oversampler;
    Rate currentRate = Rate::x1;
    FilterMode filterMode = FilterMode::FIR;

    // Pending changes (set from GUI thread, applied on audio thread)
    std::atomic<int> pendingRate{-1};      // -1 = no change pending
    std::atomic<int> pendingFilterMode{-1}; // -1 = no change pending

    double baseSampleRate = 44100.0;
    int baseBlockSize = 512;
    bool isPrepared = false;
};

//==============================================================================
// Preset Manager
//==============================================================================
class PresetManager
{
public:
    PresetManager(juce::AudioProcessorValueTreeState& apvts,
                  std::function<std::vector<std::pair<double, double>>()> getCurveFunc,
                  std::function<void(const std::vector<std::pair<double, double>>&)> setCurveFunc);

    void savePreset(const juce::String& name);
    void loadPreset(const juce::String& name);
    void deletePreset(const juce::String& name);

    juce::StringArray getPresetList() const;
    juce::File getPresetDirectory() const;

    int getCurrentPresetIndex() const { return currentPresetIndex; }
    juce::String getCurrentPresetName() const { return currentPresetName; }

private:
    juce::AudioProcessorValueTreeState& apvts;
    std::function<std::vector<std::pair<double, double>>()> getCurve;
    std::function<void(const std::vector<std::pair<double, double>>&)> setCurve;

    int currentPresetIndex = -1;
    juce::String currentPresetName;

    juce::File getPresetFile(const juce::String& name) const;
};

//==============================================================================
// Main Plugin Processor
//==============================================================================
class PhaseCorrectorAudioProcessor : public juce::AudioProcessor,
                                      public juce::AudioProcessorValueTreeState::Listener
{
public:
    PhaseCorrectorAudioProcessor();
    ~PhaseCorrectorAudioProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    void parameterChanged(const juce::String& parameterID, float newValue) override;

    // Public interface
    bool loadCSVFile(const juce::File& file);
    void setCurvePoints(const std::vector<std::pair<double, double>>& points);
    void clearCurve();
    juce::AudioProcessorValueTreeState& getAPVTS() { return apvts; }
    const std::vector<std::pair<double, double>>& getCurrentCurvePoints() const { return currentCurvePoints; }
    bool hasCurveLoaded() const { return curveLoaded.load(); }

    // Access for visualization
    const NyquistFilter& getNyquistFilter() const { return nyquistFilter; }
    const PhaseProcessor& getPhaseProcessor() const { return phaseProcessor; }
    PresetManager& getPresetManager() { return presetManager; }

private:
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    juce::AudioProcessorValueTreeState apvts;

    // DSP Components
    PhaseProcessor phaseProcessor;
    OversamplingManager oversamplingManager;
    NyquistFilter nyquistFilter;
    juce::dsp::Gain<float> outputGain;

    // Preset Manager
    PresetManager presetManager;

    // State
    std::vector<std::pair<double, double>> currentCurvePoints;
    std::atomic<bool> curveLoaded{false};

    double lastSampleRate = 44100.0;
    int lastBlockSize = 512;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PhaseCorrectorAudioProcessor)
};
