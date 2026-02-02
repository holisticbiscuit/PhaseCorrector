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
// Double-Precision FFT (Cooley-Tukey Radix-2)
// Custom implementation for 64-bit processing since JUCE FFT only supports float
//==============================================================================
class DoubleFFT
{
public:
    DoubleFFT() = default;
    explicit DoubleFFT(int order);

    void initialize(int order);
    void performRealForward(double* data);   // In-place, real input
    void performRealInverse(double* data);   // In-place, real output

    int getSize() const { return fftSize; }

private:
    void computeTwiddles();
    void fftCore(std::complex<double>* data, bool inverse);
    void bitReverse(std::complex<double>* data);

    int fftOrder = 0;
    int fftSize = 0;
    std::vector<std::complex<double>> twiddles;
    std::vector<std::complex<double>> workBuffer;
    std::vector<int> bitRevTable;
};

//==============================================================================
// FFT-based Phase Processor with Overlap-Add
//==============================================================================
class PhaseProcessor
{
public:
    // Quality levels - determines FFT size and frequency resolution
    // Analysis size determines latency and FFT size
    enum class Quality
    {
        Low = 0,       // 1024 FFT, ~43 Hz resolution @ 44.1k
        Medium = 1,    // 2048 FFT, ~21 Hz resolution
        High = 2,      // 4096 FFT, ~10.7 Hz resolution (default)
        VeryHigh = 3,  // 8192 FFT, ~5.4 Hz resolution
        Extreme = 4,   // 32768 FFT, ~1.35 Hz resolution
        Ultra64k = 5,  // 65536 FFT, ~0.67 Hz resolution
        Ultra128k = 6, // 131072 FFT, ~0.34 Hz resolution
        Ultra256k = 7  // 262144 FFT, ~0.17 Hz resolution
    };

    // Overlap amount - affects latency and quality
    enum class Overlap
    {
        Percent50 = 0,      // hopSize = fftSize/2, 2 overlaps
        Percent75 = 1,      // hopSize = fftSize/4, 4 overlaps
        Percent875 = 2,     // hopSize = fftSize/8, 8 overlaps
        Percent9375 = 3,    // hopSize = fftSize/16, 16 overlaps
        Percent96875 = 4,   // hopSize = fftSize/32, 32 overlaps
        Percent984375 = 5   // hopSize = fftSize/64, 64 overlaps
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

    // Get current latency in samples (at oversampled rate)
    int getLatencySamples() const { return analysisSize; }

    // Calculate expected latency for given quality
    static int getLatencyForQuality(Quality q)
    {
        switch (q)
        {
            case Quality::Low:       return 1024;
            case Quality::Medium:    return 2048;
            case Quality::High:      return 4096;
            case Quality::VeryHigh:  return 8192;
            case Quality::Extreme:   return 32768;
            case Quality::Ultra64k:  return 65536;
            case Quality::Ultra128k: return 131072;
            case Quality::Ultra256k: return 262144;
            default:                 return 4096;
        }
    }

    const std::vector<double>& getPhaseTable() const { return phaseTable; }
    double getSampleRate() const { return currentSampleRate; }
    int getFFTSize() const { return fftSize; }
    int getAnalysisSize() const { return analysisSize; }

    static juce::StringArray getQualityNames();
    static juce::StringArray getOverlapNames();

private:
    void rebuildPhaseTable();
    void rebuildImpulseResponse();  // Build all-pass filter IR from phase curve
    void processFrame(int channel);
    void processFrameBypass(int channel);  // Fast path when no phase processing needed
    void reconfigure();  // Rebuilds FFT, windows, buffers
    void buildWindows();

    DoubleFFT fft;  // 64-bit double precision FFT

    struct ChannelState
    {
        std::vector<double> inputBuffer;
        std::vector<double> outputBuffer;
        std::vector<double> fftBuffer;  // Pre-allocated FFT workspace
        int inputWritePos = 0;
        int outputReadPos = 0;
        int samplesUntilNextFrame = 0;

        void resize(int bufferSize, int newFftSize, int newHopSize)
        {
            inputBuffer.resize(bufferSize, 0.0);
            outputBuffer.resize(bufferSize, 0.0);
            fftBuffer.resize(newFftSize * 2, 0.0);  // Real + imaginary
            inputWritePos = 0;
            outputReadPos = 0;
            samplesUntilNextFrame = newHopSize;
        }

        void clear()
        {
            std::fill(inputBuffer.begin(), inputBuffer.end(), 0.0);
            std::fill(outputBuffer.begin(), outputBuffer.end(), 0.0);
            inputWritePos = 0;
            outputReadPos = 0;
        }
    };

    std::array<ChannelState, 2> channels;

    // Dynamic windows (double precision)
    std::vector<double> analysisWindow;
    std::vector<double> synthesisWindow;
    double windowCompensation = 1.0;

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

    // Phase modification table (double precision)
    std::vector<double> phaseTable;
    std::vector<double> filterIR;        // All-pass filter impulse response
    std::vector<double> filterSpectrum;  // FFT of filter IR (for fast convolution)
    std::atomic<bool> filterIRReady{false};
    CubicSpline phaseCurve;
    std::mutex curveMutex;

    double currentSampleRate = 44100.0;
    std::atomic<float> phaseDepth{1.0f};

    static constexpr double MIN_FREQ = 20.0;
    static constexpr double MAX_FREQ = 20000.0;
    static constexpr double LOG_MIN_FREQ = 1.301030;
    static constexpr double LOG_MAX_FREQ = 4.301030;
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
    const PhaseProcessor& getPhaseProcessor() const { return phaseProcessor; }
    PresetManager& getPresetManager() { return presetManager; }

private:
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    juce::AudioProcessorValueTreeState apvts;

    // DSP Components
    PhaseProcessor phaseProcessor;
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
