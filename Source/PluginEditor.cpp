/*
  ==============================================================================
    PhaseCorrector - Plugin Editor Implementation
    Resizable UI with Nyquist filter visualization
  ==============================================================================
*/

#include "PluginEditor.h"

//==============================================================================
// Nyquist Filter Visualizer Implementation
//==============================================================================
NyquistVisualizer::NyquistVisualizer(PhaseCorrectorAudioProcessor& processor)
    : audioProcessor(processor)
{
    startTimerHz(30);
}

NyquistVisualizer::~NyquistVisualizer()
{
    stopTimer();
}

void NyquistVisualizer::timerCallback()
{
    updateResponsePath();
    repaint();
}

float NyquistVisualizer::freqToX(float freq) const
{
    float logMin = std::log10(MIN_FREQ);
    float logMax = std::log10(MAX_FREQ);
    float logFreq = std::log10(std::max(freq, MIN_FREQ));
    return (logFreq - logMin) / (logMax - logMin) * static_cast<float>(getWidth());
}

float NyquistVisualizer::dbToY(float db) const
{
    float normalized = (db - MIN_DB) / (MAX_DB - MIN_DB);
    return static_cast<float>(getHeight()) * (1.0f - normalized);
}

void NyquistVisualizer::updateResponsePath()
{
    responsePath.clear();

    const auto& filter = audioProcessor.getNyquistFilter();
    bool started = false;

    for (int x = 0; x < getWidth(); ++x)
    {
        float logMin = std::log10(MIN_FREQ);
        float logMax = std::log10(MAX_FREQ);
        float logFreq = logMin + (static_cast<float>(x) / getWidth()) * (logMax - logMin);
        float freq = std::pow(10.0f, logFreq);

        float mag = filter.getMagnitudeAtFrequency(freq);
        float db = (mag > 0.0f) ? 20.0f * std::log10(mag) : MIN_DB;
        db = juce::jlimit(MIN_DB, MAX_DB, db);

        float y = dbToY(db);

        if (!started)
        {
            responsePath.startNewSubPath(static_cast<float>(x), y);
            started = true;
        }
        else
        {
            responsePath.lineTo(static_cast<float>(x), y);
        }
    }
}

void NyquistVisualizer::paint(juce::Graphics& g)
{
    // Background
    g.setColour(juce::Colour(0xff1a1a2e));
    g.fillRoundedRectangle(getLocalBounds().toFloat(), 4.0f);

    // Grid lines
    g.setColour(juce::Colour(0xff2a2a4e));

    // Frequency grid
    std::array<float, 5> freqs = { 100.0f, 1000.0f, 5000.0f, 10000.0f, 20000.0f };
    for (float freq : freqs)
    {
        float x = freqToX(freq);
        g.drawVerticalLine(static_cast<int>(x), 0.0f, static_cast<float>(getHeight()));
    }

    // dB grid
    std::array<float, 5> dbs = { 0.0f, -12.0f, -24.0f, -36.0f, -48.0f };
    for (float db : dbs)
    {
        float y = dbToY(db);
        if (db == 0.0f)
            g.setColour(juce::Colour(0xff4a4a7e));
        else
            g.setColour(juce::Colour(0xff2a2a4e));
        g.drawHorizontalLine(static_cast<int>(y), 0.0f, static_cast<float>(getWidth()));
    }

    // Draw response curve
    bool enabled = audioProcessor.getAPVTS().getRawParameterValue("nyquistEnabled")->load() > 0.5f;

    if (enabled)
    {
        // Glow effect
        g.setColour(juce::Colour(0xffff6b6b).withAlpha(0.3f));
        g.strokePath(responsePath, juce::PathStrokeType(4.0f));

        g.setColour(juce::Colour(0xffff6b6b).withAlpha(0.6f));
        g.strokePath(responsePath, juce::PathStrokeType(2.0f));

        g.setColour(juce::Colour(0xffff6b6b));
        g.strokePath(responsePath, juce::PathStrokeType(1.0f));
    }
    else
    {
        g.setColour(juce::Colour(0xff666666));
        g.strokePath(responsePath, juce::PathStrokeType(1.0f));
    }

    // Border
    g.setColour(juce::Colour(0xff4a4a7e));
    g.drawRoundedRectangle(getLocalBounds().toFloat().reduced(0.5f), 4.0f, 1.0f);

    // Labels
    g.setColour(juce::Colours::grey);
    g.setFont(9.0f);
    g.drawText("0dB", 2, static_cast<int>(dbToY(0.0f)) - 6, 30, 12, juce::Justification::left);
    g.drawText("-24", 2, static_cast<int>(dbToY(-24.0f)) - 6, 30, 12, juce::Justification::left);
    g.drawText("-48", 2, static_cast<int>(dbToY(-48.0f)) - 6, 30, 12, juce::Justification::left);
}

void NyquistVisualizer::resized()
{
    updateResponsePath();
}

//==============================================================================
// Phase Curve Display Implementation with Drawing Support
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
    // Sync with processor if we're not currently drawing
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

    // Clamp to valid range
    logFreq = juce::jlimit(LOG_MIN_FREQ, LOG_MAX_FREQ, logFreq);
    phase = juce::jlimit(-1.0f, 1.0f, phase);

    // Check minimum spacing from last drawn point for smooth curves
    if (lastDrawX >= 0.0f)
    {
        float lastLogFreq = xToLogFreq(lastDrawX);
        if (std::abs(logFreq - lastLogFreq) < POINT_SPACING)
            return;
    }

    // Find insertion point to maintain sorted order
    auto it = std::lower_bound(editablePoints.begin(), editablePoints.end(),
                                std::make_pair(static_cast<double>(logFreq), 0.0),
                                [](const auto& a, const auto& b) { return a.first < b.first; });

    // Check if we should update existing point or insert new one
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
    // Sort points by frequency
    std::sort(editablePoints.begin(), editablePoints.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Update the processor directly
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
    // Sync editablePoints from the processor's curve
    const auto& processorPoints = audioProcessor.getCurrentCurvePoints();
    editablePoints.clear();
    editablePoints.reserve(processorPoints.size());
    for (const auto& pt : processorPoints)
        editablePoints.push_back(pt);
    repaint();
}

void PhaseCurveDisplay::invertCurve()
{
    // Flip all phase values
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

        // Check if clicking within graph area
        if (x >= marginLeft && x <= getWidth() - marginRight &&
            y >= marginTop && y <= getHeight() - marginBottom)
        {
            // Check if clicking on existing point
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
        // Right-click to delete nearest point
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
        // Interpolate points for smooth drawing
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
    // Double-click to add a single point precisely
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

    g.setColour(juce::Colour(0xff1a1a2e));
    g.fillRoundedRectangle(graphArea.toFloat(), 5.0f);

    g.setColour(juce::Colour(0xff2a2a4e));

    // Frequency grid
    std::array<float, 9> freqs = { 20.0f, 50.0f, 100.0f, 200.0f, 500.0f, 1000.0f, 2000.0f, 5000.0f, 10000.0f };
    for (float freq : freqs)
    {
        float x = logFreqToX(std::log10(freq));
        g.drawVerticalLine(static_cast<int>(x), static_cast<float>(marginTop),
                           static_cast<float>(getHeight() - marginBottom));
    }

    // Phase grid
    std::array<float, 5> phases = { -1.0f, -0.5f, 0.0f, 0.5f, 1.0f };
    for (float phase : phases)
    {
        float y = phaseToY(phase);
        if (std::abs(phase) < 0.01f)
            g.setColour(juce::Colour(0xff4a4a7e));
        else
            g.setColour(juce::Colour(0xff2a2a4e));
        g.drawHorizontalLine(static_cast<int>(y), static_cast<float>(marginLeft),
                             static_cast<float>(getWidth() - marginRight));
    }

    g.setColour(juce::Colour(0xff4a4a7e));
    g.drawRoundedRectangle(graphArea.toFloat(), 5.0f, 1.0f);
}

void PhaseCurveDisplay::drawFrequencyLabels(juce::Graphics& g)
{
    g.setColour(juce::Colours::lightgrey);
    g.setFont(static_cast<float>(getWidth()) / 80.0f);

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
    g.setColour(juce::Colours::lightgrey);
    g.setFont(static_cast<float>(getWidth()) / 80.0f);

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
    // Use editable points if available, otherwise processor points
    const auto& points = editablePoints.empty() ? audioProcessor.getCurrentCurvePoints() : editablePoints;

    if (points.empty())
    {
        g.setColour(juce::Colours::grey);
        g.setFont(static_cast<float>(getWidth()) / 50.0f);
        g.drawText("Click and drag to draw phase curve, or drop CSV file",
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

    // Glow effect
    g.setColour(juce::Colour(0xff00ffaa).withAlpha(0.3f));
    g.strokePath(curvePath, juce::PathStrokeType(5.0f, juce::PathStrokeType::curved));

    g.setColour(juce::Colour(0xff00ffaa).withAlpha(0.5f));
    g.strokePath(curvePath, juce::PathStrokeType(3.0f, juce::PathStrokeType::curved));

    g.setColour(juce::Colour(0xff00ffaa));
    g.strokePath(curvePath, juce::PathStrokeType(1.5f, juce::PathStrokeType::curved));
}

void PhaseCurveDisplay::drawPoints(juce::Graphics& g)
{
    // Draw individual points when there are few enough to see
    if (editablePoints.size() > 0 && editablePoints.size() < 200)
    {
        for (const auto& point : editablePoints)
        {
            float x = logFreqToX(static_cast<float>(point.first));
            float y = phaseToY(static_cast<float>(point.second));

            g.setColour(juce::Colour(0xff00ffaa).withAlpha(0.8f));
            g.fillEllipse(x - 3.0f, y - 3.0f, 6.0f, 6.0f);
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

    // Show point count
    const auto& points = editablePoints.empty() ? audioProcessor.getCurrentCurvePoints() : editablePoints;
    if (!points.empty())
    {
        g.setColour(juce::Colours::grey);
        g.setFont(10.0f);
        g.drawText(juce::String(points.size()) + " points",
                   getWidth() - marginRight - 80, marginTop + 5, 75, 15,
                   juce::Justification::right);
    }

    if (isDraggingOver)
    {
        auto graphArea = getLocalBounds()
            .withTrimmedLeft(marginLeft).withTrimmedRight(marginRight)
            .withTrimmedTop(marginTop).withTrimmedBottom(marginBottom);

        g.setColour(juce::Colour(0xff00ffaa).withAlpha(0.2f));
        g.fillRoundedRectangle(graphArea.toFloat(), 5.0f);

        g.setColour(juce::Colour(0xff00ffaa));
        g.setFont(16.0f);
        g.drawText("Drop CSV file here", getLocalBounds(), juce::Justification::centred);
    }

    // Drawing indicator
    if (isDrawing)
    {
        g.setColour(juce::Colour(0xffff6b6b));
        g.setFont(10.0f);
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
                // Sync editable points with loaded curve
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
// Look and Feel Implementation
//==============================================================================
PhaseCorrectorLookAndFeel::PhaseCorrectorLookAndFeel()
{
    accentColour = juce::Colour(0xff00ffaa);
    backgroundColour = juce::Colour(0xff16162a);

    setColour(juce::Slider::rotarySliderFillColourId, accentColour);
    setColour(juce::Slider::rotarySliderOutlineColourId, juce::Colour(0xff2a2a4e));
    setColour(juce::ComboBox::backgroundColourId, juce::Colour(0xff1a1a2e));
    setColour(juce::ComboBox::textColourId, juce::Colours::white);
    setColour(juce::ComboBox::outlineColourId, juce::Colour(0xff4a4a7e));
    setColour(juce::ComboBox::arrowColourId, accentColour);
    setColour(juce::PopupMenu::backgroundColourId, juce::Colour(0xff1a1a2e));
    setColour(juce::PopupMenu::textColourId, juce::Colours::white);
    setColour(juce::PopupMenu::highlightedBackgroundColourId, accentColour.withAlpha(0.3f));
    setColour(juce::Label::textColourId, juce::Colours::white);
}

void PhaseCorrectorLookAndFeel::drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
                                                  float sliderPos, float rotaryStartAngle, float rotaryEndAngle,
                                                  juce::Slider& slider)
{
    const float radius = static_cast<float>(juce::jmin(width, height)) / 2.0f - 4.0f;
    const float centreX = static_cast<float>(x) + static_cast<float>(width) * 0.5f;
    const float centreY = static_cast<float>(y) + static_cast<float>(height) * 0.5f;
    const float angle = rotaryStartAngle + sliderPos * (rotaryEndAngle - rotaryStartAngle);

    // Background arc
    juce::Path backgroundArc;
    backgroundArc.addCentredArc(centreX, centreY, radius, radius, 0.0f,
                                 rotaryStartAngle, rotaryEndAngle, true);
    g.setColour(juce::Colour(0xff2a2a4e));
    g.strokePath(backgroundArc, juce::PathStrokeType(4.0f, juce::PathStrokeType::curved,
                                                      juce::PathStrokeType::rounded));

    // Value arc
    juce::Path valueArc;
    valueArc.addCentredArc(centreX, centreY, radius, radius, 0.0f,
                           rotaryStartAngle, angle, true);
    g.setColour(accentColour);
    g.strokePath(valueArc, juce::PathStrokeType(4.0f, juce::PathStrokeType::curved,
                                                 juce::PathStrokeType::rounded));

    // Pointer
    juce::Path pointer;
    const float pointerLength = radius * 0.6f;
    const float pointerThickness = 3.0f;
    pointer.addRoundedRectangle(-pointerThickness * 0.5f, -radius + 4.0f,
                                 pointerThickness, pointerLength, 1.0f);
    g.setColour(juce::Colours::white);
    g.fillPath(pointer, juce::AffineTransform::rotation(angle).translated(centreX, centreY));

    // Center
    g.setColour(juce::Colour(0xff1a1a2e));
    g.fillEllipse(centreX - 8.0f, centreY - 8.0f, 16.0f, 16.0f);
    g.setColour(juce::Colour(0xff4a4a7e));
    g.drawEllipse(centreX - 8.0f, centreY - 8.0f, 16.0f, 16.0f, 1.0f);

    // Value text
    g.setColour(juce::Colours::white);
    g.setFont(std::max(9.0f, static_cast<float>(width) / 7.0f));
    juce::String valueText = slider.getTextFromValue(slider.getValue());
    g.drawText(valueText, x, static_cast<int>(centreY + radius * 0.5f),
               width, 15, juce::Justification::centred, false);
}

void PhaseCorrectorLookAndFeel::drawComboBox(juce::Graphics& g, int width, int height, bool,
                                              int, int, int, int, juce::ComboBox& box)
{
    g.setColour(box.findColour(juce::ComboBox::backgroundColourId));
    g.fillRoundedRectangle(0, 0, static_cast<float>(width), static_cast<float>(height), 4.0f);

    g.setColour(box.findColour(juce::ComboBox::outlineColourId));
    g.drawRoundedRectangle(0.5f, 0.5f, static_cast<float>(width) - 1.0f,
                            static_cast<float>(height) - 1.0f, 4.0f, 1.0f);

    juce::Rectangle<int> arrowZone(width - 25, 0, 20, height);
    juce::Path path;
    path.startNewSubPath(static_cast<float>(arrowZone.getX()) + 3.0f,
                         static_cast<float>(arrowZone.getCentreY()) - 2.0f);
    path.lineTo(static_cast<float>(arrowZone.getCentreX()),
                static_cast<float>(arrowZone.getCentreY()) + 3.0f);
    path.lineTo(static_cast<float>(arrowZone.getRight()) - 3.0f,
                static_cast<float>(arrowZone.getCentreY()) - 2.0f);

    g.setColour(box.findColour(juce::ComboBox::arrowColourId));
    g.strokePath(path, juce::PathStrokeType(2.0f));
}

void PhaseCorrectorLookAndFeel::drawToggleButton(juce::Graphics& g, juce::ToggleButton& button,
                                                  bool shouldDrawButtonAsHighlighted, bool)
{
    auto bounds = button.getLocalBounds().toFloat().reduced(2.0f);

    g.setColour(button.getToggleState() ? accentColour.withAlpha(0.3f) : juce::Colour(0xff1a1a2e));
    g.fillRoundedRectangle(bounds, 4.0f);

    g.setColour(button.getToggleState() ? accentColour : juce::Colour(0xff4a4a7e));
    g.drawRoundedRectangle(bounds, 4.0f, shouldDrawButtonAsHighlighted ? 2.0f : 1.0f);

    g.setColour(button.getToggleState() ? juce::Colours::white : juce::Colours::grey);
    g.setFont(std::max(10.0f, bounds.getHeight() * 0.5f));
    g.drawText(button.getButtonText(), bounds, juce::Justification::centred);
}

//==============================================================================
// Main Editor Implementation
//==============================================================================
PhaseCorrectorAudioProcessorEditor::PhaseCorrectorAudioProcessorEditor(PhaseCorrectorAudioProcessor& p)
    : AudioProcessorEditor(&p),
      audioProcessor(p),
      curveDisplay(p),
      nyquistVisualizer(p)
{
    setLookAndFeel(&lookAndFeel);

    // Enable resizing
    setResizable(true, true);
    setResizeLimits(MIN_WIDTH, MIN_HEIGHT, MAX_WIDTH, MAX_HEIGHT);
    getConstrainer()->setFixedAspectRatio(static_cast<double>(BASE_WIDTH) / BASE_HEIGHT);

    // Title
    titleLabel.setText("PhaseCorrector", juce::dontSendNotification);
    titleLabel.setFont(juce::FontOptions(24.0f, juce::Font::bold));
    titleLabel.setColour(juce::Label::textColourId, juce::Colour(0xff00ffaa));
    titleLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(titleLabel);

    // Status label
    statusLabel.setText("Low-THD Freeform Phase EQ | 64-bit Processing", juce::dontSendNotification);
    statusLabel.setFont(juce::FontOptions(11.0f));
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::grey);
    statusLabel.setJustificationType(juce::Justification::centredRight);
    addAndMakeVisible(statusLabel);

    // Displays
    addAndMakeVisible(curveDisplay);
    addAndMakeVisible(nyquistVisualizer);

    // Clear button
    clearButton.setButtonText("Clear");
    clearButton.onClick = [this]() { curveDisplay.clearCurve(); };
    addAndMakeVisible(clearButton);

    // Invert button (flips phase curve for correction mode)
    invertButton.setButtonText("Invert");
    invertButton.onClick = [this]() { curveDisplay.invertCurve(); };
    addAndMakeVisible(invertButton);

    // Preset controls
    presetBox.setTextWhenNothingSelected("Select Preset...");
    presetBox.onChange = [this]()
    {
        if (presetBox.getSelectedId() > 0)
        {
            auto presetName = presetBox.getText();
            audioProcessor.getPresetManager().loadPreset(presetName);
            // Sync the curve display with loaded curve
            curveDisplay.loadFromProcessor();
        }
    };
    addAndMakeVisible(presetBox);
    refreshPresetList();

    savePresetButton.setButtonText("Save");
    savePresetButton.onClick = [this]()
    {
        // Always show input dialog for preset name
        auto* alertWindow = new juce::AlertWindow("Save Preset", "Enter preset name:", juce::MessageBoxIconType::NoIcon);

        // Pre-fill with current preset name if one is selected
        auto currentName = presetBox.getText();
        if (currentName == "Select Preset..." || currentName.isEmpty())
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

    deletePresetButton.setButtonText("Del");
    deletePresetButton.onClick = [this]()
    {
        auto name = presetBox.getText();
        if (name.isNotEmpty() && name != "Select Preset...")
        {
            audioProcessor.getPresetManager().deletePreset(name);
            refreshPresetList();
        }
    };
    addAndMakeVisible(deletePresetButton);

    // Main controls
    setupLabel(fftQualityLabel, "FFT Quality");
    fftQualityBox.addItemList(PhaseProcessor::getQualityNames(), 1);
    addAndMakeVisible(fftQualityBox);
    fftQualityAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        audioProcessor.getAPVTS(), "fftQuality", fftQualityBox);

    setupLabel(fftOverlapLabel, "Overlap");
    fftOverlapBox.addItemList(PhaseProcessor::getOverlapNames(), 1);
    addAndMakeVisible(fftOverlapBox);
    fftOverlapAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        audioProcessor.getAPVTS(), "fftOverlap", fftOverlapBox);

    setupLabel(depthLabel, "Depth");
    setupSlider(depthSlider, "%");
    depthAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getAPVTS(), "depth", depthSlider);

    setupLabel(dryWetLabel, "Dry/Wet");
    setupSlider(dryWetSlider, "%");
    dryWetAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getAPVTS(), "dryWet", dryWetSlider);

    setupLabel(outputGainLabel, "Output");
    setupSlider(outputGainSlider, "dB");
    outputGainAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getAPVTS(), "outputGain", outputGainSlider);

    // Nyquist controls
    setupLabel(nyquistLabel, "Nyquist Filter");
    nyquistEnableButton.setButtonText("Enable");
    addAndMakeVisible(nyquistEnableButton);
    nyquistEnableAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        audioProcessor.getAPVTS(), "nyquistEnabled", nyquistEnableButton);

    setupLabel(nyquistFreqLabel, "Frequency");
    setupSlider(nyquistFreqSlider, "Hz");
    nyquistFreqSlider.setSkewFactorFromMidPoint(5000.0);
    nyquistFreqAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getAPVTS(), "nyquistFreq", nyquistFreqSlider);

    setupLabel(nyquistQLabel, "Q");
    setupSlider(nyquistQSlider, "");
    nyquistQAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getAPVTS(), "nyquistQ", nyquistQSlider);

    setupLabel(nyquistSlopeLabel, "Slope");
    nyquistSlopeBox.addItemList({ "12dB", "24dB", "36dB", "48dB", "60dB", "72dB", "84dB", "96dB" }, 1);
    addAndMakeVisible(nyquistSlopeBox);
    nyquistSlopeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        audioProcessor.getAPVTS(), "nyquistSlope", nyquistSlopeBox);

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
    label.setFont(juce::FontOptions(11.0f));
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
    juce::ColourGradient gradient(juce::Colour(0xff0f0f1a), 0.0f, 0.0f,
                                   juce::Colour(0xff16162a), 0.0f, static_cast<float>(getHeight()),
                                   false);
    g.setGradientFill(gradient);
    g.fillAll();

    // Control panels background
    float scale = static_cast<float>(getWidth()) / BASE_WIDTH;
    int controlsY = static_cast<int>(410 * scale);

    g.setColour(juce::Colour(0xff12121f));
    g.fillRect(0, controlsY, getWidth(), getHeight() - controlsY);

    g.setColour(juce::Colour(0xff4a4a7e));
    g.drawHorizontalLine(controlsY, 0.0f, static_cast<float>(getWidth()));

    // Nyquist section separator
    int nyquistX = static_cast<int>(480 * scale);
    g.setColour(juce::Colour(0xff2a2a4e));
    g.drawVerticalLine(nyquistX, static_cast<float>(controlsY + 10),
                       static_cast<float>(getHeight() - 10));
}

void PhaseCorrectorAudioProcessorEditor::resized()
{
    float scale = static_cast<float>(getWidth()) / BASE_WIDTH;

    // Header
    int headerH = static_cast<int>(45 * scale);
    titleLabel.setBounds(static_cast<int>(15 * scale), 0,
                         static_cast<int>(200 * scale), headerH);

    // Clear and Invert buttons in header
    clearButton.setBounds(static_cast<int>(220 * scale), static_cast<int>(10 * scale),
                          static_cast<int>(50 * scale), static_cast<int>(25 * scale));
    invertButton.setBounds(static_cast<int>(275 * scale), static_cast<int>(10 * scale),
                           static_cast<int>(50 * scale), static_cast<int>(25 * scale));

    // Preset controls in header
    presetBox.setBounds(static_cast<int>(340 * scale), static_cast<int>(10 * scale),
                        static_cast<int>(140 * scale), static_cast<int>(25 * scale));
    savePresetButton.setBounds(static_cast<int>(485 * scale), static_cast<int>(10 * scale),
                               static_cast<int>(40 * scale), static_cast<int>(25 * scale));
    deletePresetButton.setBounds(static_cast<int>(530 * scale), static_cast<int>(10 * scale),
                                 static_cast<int>(35 * scale), static_cast<int>(25 * scale));

    statusLabel.setBounds(getWidth() - static_cast<int>(280 * scale), 0,
                          static_cast<int>(270 * scale), headerH);

    // Phase curve display (main area)
    int curveY = headerH;
    int curveH = static_cast<int>(280 * scale);
    curveDisplay.setBounds(static_cast<int>(10 * scale), curveY,
                           getWidth() - static_cast<int>(20 * scale), curveH);

    // Nyquist visualizer (below phase curve)
    int nyqVisY = curveY + curveH + static_cast<int>(5 * scale);
    int nyqVisH = static_cast<int>(75 * scale);
    nyquistVisualizer.setBounds(static_cast<int>(10 * scale), nyqVisY,
                                getWidth() - static_cast<int>(20 * scale), nyqVisH);

    // Controls area
    int controlsY = nyqVisY + nyqVisH + static_cast<int>(10 * scale);
    int knobSize = static_cast<int>(70 * scale);
    int labelH = static_cast<int>(15 * scale);
    int spacing = static_cast<int>(15 * scale);

    // Main controls (left side) - FFT Quality and Overlap combo boxes
    int x = static_cast<int>(15 * scale);
    int y = controlsY + static_cast<int>(5 * scale);
    int comboW = static_cast<int>(100 * scale);
    int comboH = static_cast<int>(22 * scale);
    int comboSpacing = static_cast<int>(10 * scale);

    // FFT Quality
    fftQualityLabel.setBounds(x, y, comboW, labelH);
    fftQualityBox.setBounds(x, y + labelH + 2, comboW, comboH);

    x += comboW + comboSpacing;

    // Overlap
    fftOverlapLabel.setBounds(x, y, comboW, labelH);
    fftOverlapBox.setBounds(x, y + labelH + 2, comboW, comboH);

    // Knobs start after the combo boxes
    x += comboW + comboSpacing;

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

    // Nyquist controls (right side)
    int nyquistX = static_cast<int>(500 * scale);
    x = nyquistX;

    // Nyquist label and enable
    nyquistLabel.setBounds(x, y, static_cast<int>(100 * scale), labelH);
    nyquistEnableButton.setBounds(x, y + labelH + 2, static_cast<int>(70 * scale),
                                  static_cast<int>(22 * scale));

    x += static_cast<int>(80 * scale);

    // Frequency
    nyquistFreqLabel.setBounds(x, y, knobSize, labelH);
    nyquistFreqSlider.setBounds(x, y + labelH, knobSize, knobSize);

    x += knobSize + spacing;

    // Q
    nyquistQLabel.setBounds(x, y, knobSize, labelH);
    nyquistQSlider.setBounds(x, y + labelH, knobSize, knobSize);

    x += knobSize + spacing;

    // Slope
    nyquistSlopeLabel.setBounds(x, y, static_cast<int>(70 * scale), labelH);
    nyquistSlopeBox.setBounds(x, y + labelH + static_cast<int>(20 * scale),
                              static_cast<int>(70 * scale), static_cast<int>(25 * scale));
}
