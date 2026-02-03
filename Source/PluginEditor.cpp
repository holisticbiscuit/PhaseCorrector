/*
  ==============================================================================
    PhaseCorrector - Plugin Editor Implementation
    Premium Liquid Metal UI - Luxury Edition
  ==============================================================================
*/

#include "PluginEditor.h"

//==============================================================================
// Premium Color Palette
//==============================================================================
namespace LiquidMetal
{
    // Base colors - deep obsidian with blue undertones
    const juce::Colour background1     = juce::Colour(0xff08080c);
    const juce::Colour background2     = juce::Colour(0xff0c0c12);
    const juce::Colour surface         = juce::Colour(0xff101018);
    const juce::Colour surfaceLight    = juce::Colour(0xff18182a);

    // Chrome/Platinum accents
    const juce::Colour chrome          = juce::Colour(0xffc8c8d0);
    const juce::Colour chromeBright    = juce::Colour(0xffe8e8f0);
    const juce::Colour chromeDim       = juce::Colour(0xff909098);
    const juce::Colour chromeDark      = juce::Colour(0xff606068);

    // Gold accent for luxury touch
    const juce::Colour gold            = juce::Colour(0xffd4af37);
    const juce::Colour goldDim         = juce::Colour(0xff8b7355);

    // Functional colors
    const juce::Colour textPrimary     = juce::Colour(0xffdcdce0);
    const juce::Colour textSecondary   = juce::Colour(0xff8888a0);
    const juce::Colour border          = juce::Colour(0xff2a2a3a);
    const juce::Colour borderLight     = juce::Colour(0xff3a3a4a);

    // Graph colors
    const juce::Colour graphBg         = juce::Colour(0xff0a0a10);
    const juce::Colour graphGrid       = juce::Colour(0xff1a1a28);
    const juce::Colour graphGridBright = juce::Colour(0xff252538);
}

//==============================================================================
// Phase Curve Display Implementation
//==============================================================================
PhaseCurveDisplay::PhaseCurveDisplay(PhaseCorrectorAudioProcessor& processor)
    : audioProcessor(processor)
{
    startTimerHz(30);
    setMouseCursor(juce::MouseCursor::CrosshairCursor);
}

PhaseCurveDisplay::~PhaseCurveDisplay()
{
    stopTimer();
}

void PhaseCurveDisplay::timerCallback()
{
    if (!isDrawing && !isDraggingPoint)
    {
        const auto& processorPoints = audioProcessor.getCurrentCurvePoints();
        if (editablePoints.empty() && !processorPoints.empty())
        {
            editablePoints = processorPoints;
        }
    }
    repaint();
}

float PhaseCurveDisplay::logFreqToX(float logFreq) const
{
    const float graphWidth = static_cast<float>(getWidth() - marginLeft - marginRight);
    const float normalized = (logFreq - LOG_MIN_FREQ) / (LOG_MAX_FREQ - LOG_MIN_FREQ);
    return static_cast<float>(marginLeft) + normalized * graphWidth;
}

float PhaseCurveDisplay::phaseToY(float normalizedPhase) const
{
    const float graphHeight = static_cast<float>(getHeight() - marginTop - marginBottom);
    const float normalized = (1.0f - normalizedPhase) / 2.0f;
    return static_cast<float>(marginTop) + normalized * graphHeight;
}

float PhaseCurveDisplay::xToLogFreq(float x) const
{
    const float graphWidth = static_cast<float>(getWidth() - marginLeft - marginRight);
    const float normalized = (x - static_cast<float>(marginLeft)) / graphWidth;
    return LOG_MIN_FREQ + normalized * (LOG_MAX_FREQ - LOG_MIN_FREQ);
}

float PhaseCurveDisplay::yToPhase(float y) const
{
    const float graphHeight = static_cast<float>(getHeight() - marginTop - marginBottom);
    const float normalized = (y - static_cast<float>(marginTop)) / graphHeight;
    return 1.0f - 2.0f * normalized;
}

int PhaseCurveDisplay::findNearestPointIndex(float x, float y, float maxDistance) const
{
    int nearestIdx = -1;
    float nearestDist = maxDistance * maxDistance;

    for (size_t i = 0; i < editablePoints.size(); ++i)
    {
        float px = logFreqToX(static_cast<float>(editablePoints[i].first));
        float py = phaseToY(static_cast<float>(editablePoints[i].second));
        float dist = (px - x) * (px - x) + (py - y) * (py - y);

        if (dist < nearestDist)
        {
            nearestDist = dist;
            nearestIdx = static_cast<int>(i);
        }
    }

    return nearestIdx;
}

void PhaseCurveDisplay::addOrUpdatePoint(float x, float y)
{
    float logFreq = xToLogFreq(x);
    float phase = yToPhase(y);

    logFreq = juce::jlimit(LOG_MIN_FREQ, LOG_MAX_FREQ, logFreq);
    phase = juce::jlimit(-1.0f, 1.0f, phase);

    if (lastDrawX >= 0.0f)
    {
        float lastLogFreq = xToLogFreq(lastDrawX);
        if (std::abs(logFreq - lastLogFreq) < POINT_SPACING)
            return;
    }

    auto it = std::lower_bound(editablePoints.begin(), editablePoints.end(),
                                std::make_pair(static_cast<double>(logFreq), 0.0),
                                [](const auto& a, const auto& b) { return a.first < b.first; });

    if (it != editablePoints.end() && std::abs(it->first - logFreq) < POINT_SPACING)
    {
        it->second = phase;
    }
    else if (it != editablePoints.begin())
    {
        auto prev = std::prev(it);
        if (std::abs(prev->first - logFreq) < POINT_SPACING)
        {
            prev->second = phase;
        }
        else if (editablePoints.size() < MAX_POINTS)
        {
            editablePoints.insert(it, std::make_pair(static_cast<double>(logFreq),
                                                      static_cast<double>(phase)));
        }
    }
    else if (editablePoints.size() < MAX_POINTS)
    {
        editablePoints.insert(it, std::make_pair(static_cast<double>(logFreq),
                                                  static_cast<double>(phase)));
    }

    lastDrawX = x;
}

void PhaseCurveDisplay::updateProcessorCurve()
{
    std::sort(editablePoints.begin(), editablePoints.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    audioProcessor.setCurvePoints(editablePoints);
}

void PhaseCurveDisplay::clearCurve()
{
    editablePoints.clear();
    audioProcessor.clearCurve();
    repaint();
}

void PhaseCurveDisplay::loadFromProcessor()
{
    const auto& processorPoints = audioProcessor.getCurrentCurvePoints();
    editablePoints.clear();
    editablePoints.reserve(processorPoints.size());
    for (const auto& pt : processorPoints)
        editablePoints.push_back(pt);
    repaint();
}

void PhaseCurveDisplay::invertCurve()
{
    for (auto& point : editablePoints)
        point.second = -point.second;
    updateProcessorCurve();
    repaint();
}

void PhaseCurveDisplay::mouseDown(const juce::MouseEvent& event)
{
    if (event.mods.isLeftButtonDown())
    {
        float x = static_cast<float>(event.x);
        float y = static_cast<float>(event.y);

        if (x >= marginLeft && x <= getWidth() - marginRight &&
            y >= marginTop && y <= getHeight() - marginBottom)
        {
            draggedPointIndex = findNearestPointIndex(x, y, 8.0f);

            if (draggedPointIndex >= 0)
            {
                isDraggingPoint = true;
            }
            else
            {
                isDrawing = true;
                lastDrawX = -1.0f;
                addOrUpdatePoint(x, y);
            }
        }
    }
    else if (event.mods.isRightButtonDown())
    {
        float x = static_cast<float>(event.x);
        float y = static_cast<float>(event.y);
        int idx = findNearestPointIndex(x, y, 15.0f);

        if (idx >= 0)
        {
            editablePoints.erase(editablePoints.begin() + idx);
            updateProcessorCurve();
        }
    }
}

void PhaseCurveDisplay::mouseDrag(const juce::MouseEvent& event)
{
    float x = juce::jlimit(static_cast<float>(marginLeft),
                           static_cast<float>(getWidth() - marginRight),
                           static_cast<float>(event.x));
    float y = juce::jlimit(static_cast<float>(marginTop),
                           static_cast<float>(getHeight() - marginBottom),
                           static_cast<float>(event.y));

    if (isDrawing)
    {
        if (lastDrawX >= 0.0f)
        {
            float dx = x - lastDrawX;
            int steps = std::max(1, static_cast<int>(std::abs(dx) / 2.0f));

            for (int i = 1; i <= steps; ++i)
            {
                float t = static_cast<float>(i) / steps;
                float interpX = lastDrawX + dx * t;
                float interpY = static_cast<float>(event.y - event.getDistanceFromDragStartY()) +
                               (y - (event.y - event.getDistanceFromDragStartY())) * t;
                interpY = juce::jlimit(static_cast<float>(marginTop),
                                        static_cast<float>(getHeight() - marginBottom), interpY);
                addOrUpdatePoint(interpX, interpY);
            }
        }
        else
        {
            addOrUpdatePoint(x, y);
        }
        lastDrawX = x;
    }
    else if (isDraggingPoint && draggedPointIndex >= 0)
    {
        float logFreq = xToLogFreq(x);
        float phase = yToPhase(y);

        logFreq = juce::jlimit(LOG_MIN_FREQ, LOG_MAX_FREQ, logFreq);
        phase = juce::jlimit(-1.0f, 1.0f, phase);

        editablePoints[draggedPointIndex].first = logFreq;
        editablePoints[draggedPointIndex].second = phase;
    }

    repaint();
}

void PhaseCurveDisplay::mouseUp(const juce::MouseEvent&)
{
    if (isDrawing || isDraggingPoint)
    {
        updateProcessorCurve();
    }

    isDrawing = false;
    isDraggingPoint = false;
    draggedPointIndex = -1;
    lastDrawX = -1.0f;
}

void PhaseCurveDisplay::mouseDoubleClick(const juce::MouseEvent& event)
{
    float x = static_cast<float>(event.x);
    float y = static_cast<float>(event.y);

    if (x >= marginLeft && x <= getWidth() - marginRight &&
        y >= marginTop && y <= getHeight() - marginBottom)
    {
        float logFreq = xToLogFreq(x);
        float phase = yToPhase(y);

        logFreq = juce::jlimit(LOG_MIN_FREQ, LOG_MAX_FREQ, logFreq);
        phase = juce::jlimit(-1.0f, 1.0f, phase);

        editablePoints.push_back(std::make_pair(static_cast<double>(logFreq),
                                                 static_cast<double>(phase)));
        updateProcessorCurve();
        repaint();
    }
}

void PhaseCurveDisplay::drawGrid(juce::Graphics& g)
{
    auto graphArea = getLocalBounds()
        .withTrimmedLeft(marginLeft)
        .withTrimmedRight(marginRight)
        .withTrimmedTop(marginTop)
        .withTrimmedBottom(marginBottom);

    // Premium graph background with depth
    juce::ColourGradient bgGradient(LiquidMetal::graphBg,
                                     static_cast<float>(graphArea.getX()), static_cast<float>(graphArea.getY()),
                                     LiquidMetal::background1,
                                     static_cast<float>(graphArea.getX()), static_cast<float>(graphArea.getBottom()), false);
    g.setGradientFill(bgGradient);
    g.fillRoundedRectangle(graphArea.toFloat(), 6.0f);

    // Inner shadow effect at top
    juce::ColourGradient innerShadow(juce::Colour(0x30000000),
                                      static_cast<float>(graphArea.getX()), static_cast<float>(graphArea.getY()),
                                      juce::Colour(0x00000000),
                                      static_cast<float>(graphArea.getX()), static_cast<float>(graphArea.getY() + 20), false);
    g.setGradientFill(innerShadow);
    g.fillRoundedRectangle(graphArea.toFloat(), 6.0f);

    // Subtle top reflection
    g.setColour(juce::Colour(0x08ffffff));
    g.drawHorizontalLine(graphArea.getY() + 1, static_cast<float>(graphArea.getX() + 6), static_cast<float>(graphArea.getRight() - 6));

    // Frequency grid lines
    g.setColour(LiquidMetal::graphGrid);
    std::array<float, 9> freqs = { 20.0f, 50.0f, 100.0f, 200.0f, 500.0f, 1000.0f, 2000.0f, 5000.0f, 10000.0f };
    for (float freq : freqs)
    {
        float x = logFreqToX(std::log10(freq));
        g.drawVerticalLine(static_cast<int>(x), static_cast<float>(marginTop), static_cast<float>(getHeight() - marginBottom));
    }

    // Phase grid lines
    std::array<float, 5> phases = { -1.0f, -0.5f, 0.0f, 0.5f, 1.0f };
    for (float phase : phases)
    {
        float y = phaseToY(phase);
        if (std::abs(phase) < 0.01f)
            g.setColour(LiquidMetal::graphGridBright);
        else
            g.setColour(LiquidMetal::graphGrid);
        g.drawHorizontalLine(static_cast<int>(y), static_cast<float>(marginLeft), static_cast<float>(getWidth() - marginRight));
    }

    // Premium border with subtle glow
    g.setColour(LiquidMetal::border);
    g.drawRoundedRectangle(graphArea.toFloat(), 6.0f, 1.5f);

    // Outer subtle glow
    g.setColour(juce::Colour(0x10ffffff));
    g.drawRoundedRectangle(graphArea.toFloat().expanded(1.0f), 7.0f, 0.5f);
}

void PhaseCurveDisplay::drawFrequencyLabels(juce::Graphics& g)
{
    g.setColour(LiquidMetal::textSecondary);
    g.setFont(juce::FontOptions(static_cast<float>(getWidth()) / 75.0f));

    struct FreqLabel { float freq; const char* text; };
    std::array<FreqLabel, 7> labels = {{
        { 20.0f, "20" }, { 100.0f, "100" }, { 500.0f, "500" },
        { 1000.0f, "1k" }, { 5000.0f, "5k" }, { 10000.0f, "10k" }, { 20000.0f, "20k" }
    }};

    for (const auto& label : labels)
    {
        float x = logFreqToX(std::log10(label.freq));
        g.drawText(label.text, static_cast<int>(x) - 20, getHeight() - marginBottom + 5,
                   40, 20, juce::Justification::centredTop);
    }
}

void PhaseCurveDisplay::drawPhaseLabels(juce::Graphics& g)
{
    g.setColour(LiquidMetal::textSecondary);
    g.setFont(juce::FontOptions(static_cast<float>(getWidth()) / 75.0f));

    struct PhaseLabel { float phase; const char* text; };
    std::array<PhaseLabel, 5> labels = {{
        { 1.0f, "+360" }, { 0.5f, "+180" }, { 0.0f, "0" }, { -0.5f, "-180" }, { -1.0f, "-360" }
    }};

    for (const auto& label : labels)
    {
        float y = phaseToY(label.phase);
        g.drawText(label.text, 5, static_cast<int>(y) - 8, marginLeft - 10, 16,
                   juce::Justification::centredRight);
    }
}

void PhaseCurveDisplay::drawCurve(juce::Graphics& g)
{
    const auto& points = editablePoints.empty() ? audioProcessor.getCurrentCurvePoints() : editablePoints;

    if (points.empty())
    {
        g.setColour(LiquidMetal::textSecondary.withAlpha(0.5f));
        g.setFont(juce::FontOptions(static_cast<float>(getWidth()) / 50.0f));
        g.drawText("Draw phase curve or drop CSV file",
                   getLocalBounds(), juce::Justification::centred);
        return;
    }

    curvePath.clear();
    bool started = false;

    for (const auto& point : points)
    {
        float x = logFreqToX(static_cast<float>(point.first));
        float y = phaseToY(static_cast<float>(point.second));

        if (!started)
        {
            curvePath.startNewSubPath(x, y);
            started = true;
        }
        else
        {
            curvePath.lineTo(x, y);
        }
    }

    // Premium liquid metal curve - multiple layers for chrome effect
    // Outer glow
    g.setColour(LiquidMetal::gold.withAlpha(0.15f));
    g.strokePath(curvePath, juce::PathStrokeType(8.0f, juce::PathStrokeType::curved));

    // Dark chrome shadow
    g.setColour(LiquidMetal::chromeDark.withAlpha(0.6f));
    g.strokePath(curvePath, juce::PathStrokeType(5.0f, juce::PathStrokeType::curved));

    // Mid chrome
    g.setColour(LiquidMetal::chromeDim);
    g.strokePath(curvePath, juce::PathStrokeType(3.5f, juce::PathStrokeType::curved));

    // Bright chrome
    g.setColour(LiquidMetal::chrome);
    g.strokePath(curvePath, juce::PathStrokeType(2.0f, juce::PathStrokeType::curved));

    // Hot highlight
    g.setColour(LiquidMetal::chromeBright);
    g.strokePath(curvePath, juce::PathStrokeType(1.0f, juce::PathStrokeType::curved));
}

void PhaseCurveDisplay::drawPoints(juce::Graphics& g)
{
    if (editablePoints.size() > 0 && editablePoints.size() < 200)
    {
        for (const auto& point : editablePoints)
        {
            float x = logFreqToX(static_cast<float>(point.first));
            float y = phaseToY(static_cast<float>(point.second));

            // Luxurious point with gold accent
            g.setColour(LiquidMetal::chromeDark);
            g.fillEllipse(x - 5.0f, y - 5.0f, 10.0f, 10.0f);
            g.setColour(LiquidMetal::chrome);
            g.fillEllipse(x - 4.0f, y - 4.0f, 8.0f, 8.0f);
            g.setColour(LiquidMetal::chromeBright);
            g.fillEllipse(x - 2.0f, y - 3.0f, 4.0f, 4.0f);
        }
    }
}

void PhaseCurveDisplay::paint(juce::Graphics& g)
{
    drawGrid(g);
    drawCurve(g);
    drawPoints(g);
    drawFrequencyLabels(g);
    drawPhaseLabels(g);

    const auto& points = editablePoints.empty() ? audioProcessor.getCurrentCurvePoints() : editablePoints;
    if (!points.empty())
    {
        g.setColour(LiquidMetal::textSecondary);
        g.setFont(juce::FontOptions(10.0f));
        g.drawText(juce::String(points.size()) + " pts",
                   getWidth() - marginRight - 60, marginTop + 5, 55, 15,
                   juce::Justification::right);
    }

    if (isDraggingOver)
    {
        auto graphArea = getLocalBounds()
            .withTrimmedLeft(marginLeft).withTrimmedRight(marginRight)
            .withTrimmedTop(marginTop).withTrimmedBottom(marginBottom);

        g.setColour(LiquidMetal::gold.withAlpha(0.1f));
        g.fillRoundedRectangle(graphArea.toFloat(), 6.0f);

        g.setColour(LiquidMetal::gold);
        g.setFont(juce::FontOptions(16.0f));
        g.drawText("Drop CSV file here", getLocalBounds(), juce::Justification::centred);
    }

    if (isDrawing)
    {
        g.setColour(LiquidMetal::gold);
        g.setFont(juce::FontOptions(10.0f));
        g.drawText("Drawing...", marginLeft + 5, marginTop + 5, 60, 15, juce::Justification::left);
    }
}

void PhaseCurveDisplay::resized()
{
    float scale = static_cast<float>(getWidth()) / 600.0f;
    marginLeft = static_cast<int>(50 * scale);
    marginRight = static_cast<int>(15 * scale);
    marginTop = static_cast<int>(15 * scale);
    marginBottom = static_cast<int>(30 * scale);
}

bool PhaseCurveDisplay::isInterestedInFileDrag(const juce::StringArray& files)
{
    for (const auto& file : files)
        if (file.endsWithIgnoreCase(".csv") || file.endsWithIgnoreCase(".txt"))
            return true;
    return false;
}

void PhaseCurveDisplay::filesDropped(const juce::StringArray& files, int, int)
{
    isDraggingOver = false;

    for (const auto& filePath : files)
    {
        juce::File file(filePath);
        if (file.existsAsFile() &&
            (file.hasFileExtension(".csv") || file.hasFileExtension(".txt")))
        {
            if (audioProcessor.loadCSVFile(file))
            {
                editablePoints = audioProcessor.getCurrentCurvePoints();
                repaint();
                break;
            }
        }
    }
}

void PhaseCurveDisplay::fileDragEnter(const juce::StringArray&, int, int)
{
    isDraggingOver = true;
    repaint();
}

void PhaseCurveDisplay::fileDragExit(const juce::StringArray&)
{
    isDraggingOver = false;
    repaint();
}

//==============================================================================
// Premium Look and Feel - Liquid Metal Luxury Theme
//==============================================================================
PhaseCorrectorLookAndFeel::PhaseCorrectorLookAndFeel()
{
    accentColour = LiquidMetal::chrome;
    backgroundColour = LiquidMetal::background1;

    setColour(juce::Slider::rotarySliderFillColourId, accentColour);
    setColour(juce::Slider::rotarySliderOutlineColourId, LiquidMetal::border);
    setColour(juce::ComboBox::backgroundColourId, LiquidMetal::surface);
    setColour(juce::ComboBox::textColourId, LiquidMetal::textPrimary);
    setColour(juce::ComboBox::outlineColourId, LiquidMetal::border);
    setColour(juce::ComboBox::arrowColourId, LiquidMetal::chrome);
    setColour(juce::PopupMenu::backgroundColourId, LiquidMetal::surface);
    setColour(juce::PopupMenu::textColourId, LiquidMetal::textPrimary);
    setColour(juce::PopupMenu::highlightedBackgroundColourId, LiquidMetal::surfaceLight);
    setColour(juce::Label::textColourId, LiquidMetal::textPrimary);
    setColour(juce::TextButton::buttonColourId, LiquidMetal::surface);
    setColour(juce::TextButton::textColourOffId, LiquidMetal::textPrimary);
}

void PhaseCorrectorLookAndFeel::drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
                                                  float sliderPos, float rotaryStartAngle, float rotaryEndAngle,
                                                  juce::Slider& slider)
{
    const float radius = static_cast<float>(juce::jmin(width, height)) / 2.0f - 4.0f;
    const float centreX = static_cast<float>(x) + static_cast<float>(width) * 0.5f;
    const float centreY = static_cast<float>(y) + static_cast<float>(height) * 0.5f;
    const float angle = rotaryStartAngle + sliderPos * (rotaryEndAngle - rotaryStartAngle);

    // Outer ring - recessed look
    g.setColour(LiquidMetal::background1);
    g.fillEllipse(centreX - radius - 2, centreY - radius - 2, (radius + 2) * 2, (radius + 2) * 2);

    // Background arc track
    juce::Path backgroundArc;
    backgroundArc.addCentredArc(centreX, centreY, radius, radius, 0.0f,
                                 rotaryStartAngle, rotaryEndAngle, true);
    g.setColour(LiquidMetal::surface);
    g.strokePath(backgroundArc, juce::PathStrokeType(6.0f, juce::PathStrokeType::curved,
                                                      juce::PathStrokeType::rounded));
    g.setColour(LiquidMetal::border);
    g.strokePath(backgroundArc, juce::PathStrokeType(5.0f, juce::PathStrokeType::curved,
                                                      juce::PathStrokeType::rounded));

    // Value arc - liquid metal gradient
    juce::Path valueArc;
    valueArc.addCentredArc(centreX, centreY, radius, radius, 0.0f,
                           rotaryStartAngle, angle, true);

    // Chrome layers for liquid metal effect
    g.setColour(LiquidMetal::chromeDark);
    g.strokePath(valueArc, juce::PathStrokeType(6.0f, juce::PathStrokeType::curved,
                                                 juce::PathStrokeType::rounded));
    g.setColour(LiquidMetal::chromeDim);
    g.strokePath(valueArc, juce::PathStrokeType(4.0f, juce::PathStrokeType::curved,
                                                 juce::PathStrokeType::rounded));
    g.setColour(LiquidMetal::chrome);
    g.strokePath(valueArc, juce::PathStrokeType(2.5f, juce::PathStrokeType::curved,
                                                 juce::PathStrokeType::rounded));
    g.setColour(LiquidMetal::chromeBright);
    g.strokePath(valueArc, juce::PathStrokeType(1.0f, juce::PathStrokeType::curved,
                                                 juce::PathStrokeType::rounded));

    // Center knob - premium metallic look
    float knobRadius = radius * 0.45f;

    // Knob shadow
    g.setColour(juce::Colour(0x40000000));
    g.fillEllipse(centreX - knobRadius + 2, centreY - knobRadius + 2, knobRadius * 2, knobRadius * 2);

    // Knob base gradient
    juce::ColourGradient knobGradient(LiquidMetal::surfaceLight, centreX - knobRadius * 0.5f, centreY - knobRadius * 0.5f,
                                       LiquidMetal::background1, centreX + knobRadius * 0.5f, centreY + knobRadius * 0.5f, true);
    g.setGradientFill(knobGradient);
    g.fillEllipse(centreX - knobRadius, centreY - knobRadius, knobRadius * 2, knobRadius * 2);

    // Knob highlight
    g.setColour(juce::Colour(0x15ffffff));
    g.fillEllipse(centreX - knobRadius * 0.6f, centreY - knobRadius * 0.8f, knobRadius * 0.8f, knobRadius * 0.5f);

    // Knob border
    g.setColour(LiquidMetal::border);
    g.drawEllipse(centreX - knobRadius, centreY - knobRadius, knobRadius * 2, knobRadius * 2, 1.0f);

    // Pointer - chrome needle with gold tip
    juce::Path pointer;
    const float pointerLength = radius * 0.55f;
    const float pointerThickness = 2.5f;
    pointer.addRoundedRectangle(-pointerThickness * 0.5f, -radius + 6.0f,
                                 pointerThickness, pointerLength, 1.0f);

    g.setColour(LiquidMetal::chromeBright);
    g.fillPath(pointer, juce::AffineTransform::rotation(angle).translated(centreX, centreY));

    // Gold dot at pointer tip
    float tipX = centreX + std::sin(angle) * (radius - 10.0f);
    float tipY = centreY - std::cos(angle) * (radius - 10.0f);
    g.setColour(LiquidMetal::gold);
    g.fillEllipse(tipX - 2.5f, tipY - 2.5f, 5.0f, 5.0f);

    // Value text
    g.setColour(LiquidMetal::textPrimary);
    g.setFont(juce::FontOptions(std::max(9.0f, static_cast<float>(width) / 7.5f)));
    juce::String valueText = slider.getTextFromValue(slider.getValue());
    g.drawText(valueText, x, static_cast<int>(centreY + radius * 0.55f),
               width, 15, juce::Justification::centred, false);
}

void PhaseCorrectorLookAndFeel::drawComboBox(juce::Graphics& g, int width, int height, bool,
                                              int, int, int, int, juce::ComboBox&)
{
    // Premium combo box with depth
    juce::ColourGradient bgGradient(LiquidMetal::surfaceLight, 0.0f, 0.0f,
                                     LiquidMetal::surface, 0.0f, static_cast<float>(height), false);
    g.setGradientFill(bgGradient);
    g.fillRoundedRectangle(0, 0, static_cast<float>(width), static_cast<float>(height), 5.0f);

    // Inner top shadow
    g.setColour(juce::Colour(0x20000000));
    g.drawHorizontalLine(1, 2.0f, static_cast<float>(width) - 2.0f);

    // Bottom highlight
    g.setColour(juce::Colour(0x08ffffff));
    g.drawHorizontalLine(height - 2, 2.0f, static_cast<float>(width) - 2.0f);

    // Border
    g.setColour(LiquidMetal::border);
    g.drawRoundedRectangle(0.5f, 0.5f, static_cast<float>(width) - 1.0f,
                            static_cast<float>(height) - 1.0f, 5.0f, 1.0f);

    // Chrome arrow
    juce::Rectangle<int> arrowZone(width - 25, 0, 20, height);
    juce::Path path;
    path.startNewSubPath(static_cast<float>(arrowZone.getX()) + 3.0f,
                         static_cast<float>(arrowZone.getCentreY()) - 2.0f);
    path.lineTo(static_cast<float>(arrowZone.getCentreX()),
                static_cast<float>(arrowZone.getCentreY()) + 3.0f);
    path.lineTo(static_cast<float>(arrowZone.getRight()) - 3.0f,
                static_cast<float>(arrowZone.getCentreY()) - 2.0f);

    g.setColour(LiquidMetal::chrome);
    g.strokePath(path, juce::PathStrokeType(2.0f));
}

void PhaseCorrectorLookAndFeel::drawToggleButton(juce::Graphics& g, juce::ToggleButton& button,
                                                  bool shouldDrawButtonAsHighlighted, bool)
{
    auto bounds = button.getLocalBounds().toFloat().reduced(2.0f);

    // Premium button with depth
    if (button.getToggleState())
    {
        juce::ColourGradient activeGradient(LiquidMetal::surfaceLight, bounds.getX(), bounds.getY(),
                                             LiquidMetal::surface, bounds.getX(), bounds.getBottom(), false);
        g.setGradientFill(activeGradient);
    }
    else
    {
        juce::ColourGradient inactiveGradient(LiquidMetal::surface, bounds.getX(), bounds.getY(),
                                               LiquidMetal::background2, bounds.getX(), bounds.getBottom(), false);
        g.setGradientFill(inactiveGradient);
    }
    g.fillRoundedRectangle(bounds, 5.0f);

    // Highlight when hovered
    if (shouldDrawButtonAsHighlighted)
    {
        g.setColour(juce::Colour(0x10ffffff));
        g.fillRoundedRectangle(bounds, 5.0f);
    }

    // Border with gold accent when active
    g.setColour(button.getToggleState() ? LiquidMetal::gold.withAlpha(0.6f) : LiquidMetal::border);
    g.drawRoundedRectangle(bounds, 5.0f, 1.0f);

    // Text
    g.setColour(button.getToggleState() ? LiquidMetal::textPrimary : LiquidMetal::textSecondary);
    g.setFont(juce::FontOptions(std::max(10.0f, bounds.getHeight() * 0.45f)));
    g.drawText(button.getButtonText(), bounds, juce::Justification::centred);
}

//==============================================================================
// Main Editor Implementation
//==============================================================================
PhaseCorrectorAudioProcessorEditor::PhaseCorrectorAudioProcessorEditor(PhaseCorrectorAudioProcessor& p)
    : AudioProcessorEditor(&p),
      audioProcessor(p),
      curveDisplay(p)
{
    setLookAndFeel(&lookAndFeel);

    setResizable(true, true);
    setResizeLimits(MIN_WIDTH, MIN_HEIGHT, MAX_WIDTH, MAX_HEIGHT);
    getConstrainer()->setFixedAspectRatio(static_cast<double>(BASE_WIDTH) / BASE_HEIGHT);

    // Premium title with elegant styling
    titleLabel.setText("PHASE CORRECTOR", juce::dontSendNotification);
    titleLabel.setFont(juce::FontOptions(22.0f, juce::Font::bold));
    titleLabel.setColour(juce::Label::textColourId, LiquidMetal::chrome);
    titleLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(titleLabel);

    // Status label - refined
    statusLabel.setText("64-BIT PRECISION | ZERO NOISE FLOOR", juce::dontSendNotification);
    statusLabel.setFont(juce::FontOptions(10.0f));
    statusLabel.setColour(juce::Label::textColourId, LiquidMetal::gold.withAlpha(0.7f));
    statusLabel.setJustificationType(juce::Justification::centredRight);
    addAndMakeVisible(statusLabel);

    addAndMakeVisible(curveDisplay);

    // Clear button - luxury styled
    clearButton.setButtonText("CLEAR");
    clearButton.onClick = [this]() { curveDisplay.clearCurve(); };
    addAndMakeVisible(clearButton);

    invertButton.setButtonText("INVERT");
    invertButton.onClick = [this]() { curveDisplay.invertCurve(); };
    addAndMakeVisible(invertButton);

    // Preset controls
    presetBox.setTextWhenNothingSelected("Presets...");
    presetBox.onChange = [this]()
    {
        if (presetBox.getSelectedId() > 0)
        {
            auto presetName = presetBox.getText();
            audioProcessor.getPresetManager().loadPreset(presetName);
            curveDisplay.loadFromProcessor();
        }
    };
    addAndMakeVisible(presetBox);
    refreshPresetList();

    savePresetButton.setButtonText("SAVE");
    savePresetButton.onClick = [this]()
    {
        auto* alertWindow = new juce::AlertWindow("Save Preset", "Enter preset name:", juce::MessageBoxIconType::NoIcon);
        auto currentName = presetBox.getText();
        if (currentName == "Presets..." || currentName.isEmpty())
            currentName = "";
        alertWindow->addTextEditor("presetName", currentName, "Name:");
        alertWindow->addButton("Save", 1, juce::KeyPress(juce::KeyPress::returnKey));
        alertWindow->addButton("Cancel", 0, juce::KeyPress(juce::KeyPress::escapeKey));

        alertWindow->enterModalState(true, juce::ModalCallbackFunction::create(
            [this, alertWindow](int result)
            {
                if (result == 1)
                {
                    auto presetName = alertWindow->getTextEditorContents("presetName");
                    if (presetName.isNotEmpty())
                    {
                        audioProcessor.getPresetManager().savePreset(presetName);
                        refreshPresetList();
                    }
                }
                delete alertWindow;
            }), true);
    };
    addAndMakeVisible(savePresetButton);

    deletePresetButton.setButtonText("DEL");
    deletePresetButton.onClick = [this]()
    {
        auto name = presetBox.getText();
        if (name.isNotEmpty() && name != "Presets...")
        {
            audioProcessor.getPresetManager().deletePreset(name);
            refreshPresetList();
        }
    };
    addAndMakeVisible(deletePresetButton);

    // Main controls with premium labels
    setupLabel(fftQualityLabel, "QUALITY");
    fftQualityBox.addItemList(PhaseProcessor::getQualityNames(), 1);
    addAndMakeVisible(fftQualityBox);
    fftQualityAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        audioProcessor.getAPVTS(), "fftQuality", fftQualityBox);

    setupLabel(fftOverlapLabel, "OVERLAP");
    fftOverlapBox.addItemList(PhaseProcessor::getOverlapNames(), 1);
    addAndMakeVisible(fftOverlapBox);
    fftOverlapAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        audioProcessor.getAPVTS(), "fftOverlap", fftOverlapBox);

    setupLabel(depthLabel, "DEPTH");
    setupSlider(depthSlider, "%");
    depthAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getAPVTS(), "depth", depthSlider);

    setupLabel(dryWetLabel, "MIX");
    setupSlider(dryWetSlider, "%");
    dryWetAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getAPVTS(), "dryWet", dryWetSlider);

    setupLabel(outputGainLabel, "OUTPUT");
    setupSlider(outputGainSlider, "dB");
    outputGainAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getAPVTS(), "outputGain", outputGainSlider);

    setSize(BASE_WIDTH, BASE_HEIGHT);
}

PhaseCorrectorAudioProcessorEditor::~PhaseCorrectorAudioProcessorEditor()
{
    setLookAndFeel(nullptr);
}

void PhaseCorrectorAudioProcessorEditor::setupSlider(juce::Slider& slider, const juce::String& suffix)
{
    slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    slider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
    slider.setTextValueSuffix(suffix);
    addAndMakeVisible(slider);
}

void PhaseCorrectorAudioProcessorEditor::setupLabel(juce::Label& label, const juce::String& text)
{
    label.setText(text, juce::dontSendNotification);
    label.setFont(juce::FontOptions(10.0f, juce::Font::bold));
    label.setColour(juce::Label::textColourId, LiquidMetal::textSecondary);
    label.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(label);
}

void PhaseCorrectorAudioProcessorEditor::refreshPresetList()
{
    auto currentText = presetBox.getText();
    presetBox.clear();

    auto presets = audioProcessor.getPresetManager().getPresetList();
    int id = 1;
    int selectedId = 0;

    for (const auto& preset : presets)
    {
        presetBox.addItem(preset, id);
        if (preset == currentText)
            selectedId = id;
        id++;
    }

    if (selectedId > 0)
        presetBox.setSelectedId(selectedId, juce::dontSendNotification);
}

void PhaseCorrectorAudioProcessorEditor::paint(juce::Graphics& g)
{
    float w = static_cast<float>(getWidth());
    float h = static_cast<float>(getHeight());

    // Premium liquid metal background
    // Base: deep obsidian with subtle blue undertone
    juce::ColourGradient baseGradient(
        LiquidMetal::background1, 0.0f, 0.0f,
        LiquidMetal::background2, 0.0f, h, false);
    g.setGradientFill(baseGradient);
    g.fillAll();

    // Subtle brushed metal texture - diagonal light sweep
    juce::ColourGradient brushedMetal(
        juce::Colour(0x00000000), 0.0f, 0.0f,
        juce::Colour(0x08909098), w * 0.3f, h * 0.3f, false);
    brushedMetal.addColour(0.5, juce::Colour(0x05c0c0c8));
    brushedMetal.addColour(1.0, juce::Colour(0x00000000));
    g.setGradientFill(brushedMetal);
    g.fillAll();

    // Top edge highlight - polished metal reflection
    juce::ColourGradient topShine(
        juce::Colour(0x18ffffff), 0.0f, 0.0f,
        juce::Colour(0x00ffffff), 0.0f, h * 0.15f, false);
    g.setGradientFill(topShine);
    g.fillRect(0.0f, 0.0f, w, h * 0.15f);

    // Subtle gold accent line at top
    g.setColour(LiquidMetal::gold.withAlpha(0.3f));
    g.fillRect(0.0f, 0.0f, w, 1.0f);
    g.setColour(LiquidMetal::gold.withAlpha(0.1f));
    g.fillRect(0.0f, 1.0f, w, 1.0f);

    // Control panel area - recessed premium look
    float scale = static_cast<float>(getWidth()) / BASE_WIDTH;
    int controlsY = static_cast<int>(360 * scale);

    // Recessed panel shadow
    juce::ColourGradient panelShadow(
        juce::Colour(0x40000000), 0.0f, static_cast<float>(controlsY),
        juce::Colour(0x00000000), 0.0f, static_cast<float>(controlsY + 8), false);
    g.setGradientFill(panelShadow);
    g.fillRect(0.0f, static_cast<float>(controlsY), w, 8.0f);

    // Panel background
    juce::ColourGradient panelGradient(
        LiquidMetal::background1, 0.0f, static_cast<float>(controlsY),
        juce::Colour(0xff060608), 0.0f, h, false);
    g.setGradientFill(panelGradient);
    g.fillRect(0, controlsY + 2, getWidth(), getHeight() - controlsY - 2);

    // Panel top edge - subtle bevel
    g.setColour(LiquidMetal::border);
    g.drawHorizontalLine(controlsY, 0.0f, w);
    g.setColour(juce::Colour(0x10ffffff));
    g.drawHorizontalLine(controlsY + 3, 0.0f, w);

    // Bottom edge - subtle gold accent
    g.setColour(LiquidMetal::gold.withAlpha(0.15f));
    g.drawHorizontalLine(getHeight() - 1, 0.0f, w);
}

void PhaseCorrectorAudioProcessorEditor::resized()
{
    float scale = static_cast<float>(getWidth()) / BASE_WIDTH;

    // Header
    int headerH = static_cast<int>(45 * scale);
    titleLabel.setBounds(static_cast<int>(15 * scale), 0,
                         static_cast<int>(200 * scale), headerH);

    // Clear and Invert buttons
    clearButton.setBounds(static_cast<int>(215 * scale), static_cast<int>(10 * scale),
                          static_cast<int>(55 * scale), static_cast<int>(25 * scale));
    invertButton.setBounds(static_cast<int>(275 * scale), static_cast<int>(10 * scale),
                           static_cast<int>(55 * scale), static_cast<int>(25 * scale));

    // Preset controls
    presetBox.setBounds(static_cast<int>(345 * scale), static_cast<int>(10 * scale),
                        static_cast<int>(120 * scale), static_cast<int>(25 * scale));
    savePresetButton.setBounds(static_cast<int>(470 * scale), static_cast<int>(10 * scale),
                               static_cast<int>(45 * scale), static_cast<int>(25 * scale));
    deletePresetButton.setBounds(static_cast<int>(520 * scale), static_cast<int>(10 * scale),
                                 static_cast<int>(40 * scale), static_cast<int>(25 * scale));

    statusLabel.setBounds(getWidth() - static_cast<int>(220 * scale), 0,
                          static_cast<int>(210 * scale), headerH);

    // Phase curve display
    int curveY = headerH;
    int curveH = static_cast<int>(305 * scale);
    curveDisplay.setBounds(static_cast<int>(10 * scale), curveY,
                           getWidth() - static_cast<int>(20 * scale), curveH);

    // Controls area
    int controlsY = curveY + curveH + static_cast<int>(12 * scale);
    int knobSize = static_cast<int>(70 * scale);
    int labelH = static_cast<int>(14 * scale);
    int spacing = static_cast<int>(20 * scale);

    int totalControlsWidth = static_cast<int>(100 * scale) * 2 + static_cast<int>(10 * scale) +
                              knobSize * 3 + spacing * 2;
    int startX = (getWidth() - totalControlsWidth) / 2;

    int x = startX;
    int y = controlsY + static_cast<int>(5 * scale);
    int comboW = static_cast<int>(100 * scale);
    int comboH = static_cast<int>(24 * scale);
    int comboSpacing = static_cast<int>(10 * scale);

    // FFT Quality
    fftQualityLabel.setBounds(x, y, comboW, labelH);
    fftQualityBox.setBounds(x, y + labelH + 2, comboW, comboH);

    x += comboW + comboSpacing;

    // Overlap
    fftOverlapLabel.setBounds(x, y, comboW, labelH);
    fftOverlapBox.setBounds(x, y + labelH + 2, comboW, comboH);

    x += comboW + spacing;

    // Depth
    depthLabel.setBounds(x, y, knobSize, labelH);
    depthSlider.setBounds(x, y + labelH, knobSize, knobSize);

    x += knobSize + spacing;

    // Dry/Wet
    dryWetLabel.setBounds(x, y, knobSize, labelH);
    dryWetSlider.setBounds(x, y + labelH, knobSize, knobSize);

    x += knobSize + spacing;

    // Output
    outputGainLabel.setBounds(x, y, knobSize, labelH);
    outputGainSlider.setBounds(x, y + labelH, knobSize, knobSize);
}
