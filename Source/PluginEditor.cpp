/*
  ==============================================================================
    PhaseCorrector - Plugin Editor Implementation
    Premium UI with liquid metal aesthetic
  ==============================================================================
*/

#include "PluginEditor.h"

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

    // Liquid metal graph background - subtle gradient
    juce::ColourGradient bgGradient(juce::Colour(0xff0c0c10), graphArea.getX(), graphArea.getY(),
                                     juce::Colour(0xff080809), graphArea.getX(), graphArea.getBottom(), false);
    g.setGradientFill(bgGradient);
    g.fillRoundedRectangle(graphArea.toFloat(), 5.0f);

    // Very subtle top highlight
    g.setColour(juce::Colour(0x08ffffff));
    g.drawHorizontalLine(graphArea.getY() + 1, graphArea.getX() + 5.0f, graphArea.getRight() - 5.0f);

    g.setColour(juce::Colour(0xff1a1a20));

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
            g.setColour(juce::Colour(0xff2a2a35));  // Brighter center line
        else
            g.setColour(juce::Colour(0xff1a1a20));
        g.drawHorizontalLine(static_cast<int>(y), static_cast<float>(marginLeft),
                             static_cast<float>(getWidth() - marginRight));
    }

    // Border - subtle metallic edge
    g.setColour(juce::Colour(0xff252530));
    g.drawRoundedRectangle(graphArea.toFloat(), 5.0f, 1.0f);
}

void PhaseCurveDisplay::drawFrequencyLabels(juce::Graphics& g)
{
    g.setColour(juce::Colour(0xff808088));
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
    g.setColour(juce::Colour(0xff808088));
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
        g.setColour(juce::Colour(0xff505058));
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

    // Liquid metal curve effect - chrome/silver glow
    g.setColour(juce::Colour(0xff505058).withAlpha(0.4f));
    g.strokePath(curvePath, juce::PathStrokeType(6.0f, juce::PathStrokeType::curved));

    g.setColour(juce::Colour(0xff808088).withAlpha(0.6f));
    g.strokePath(curvePath, juce::PathStrokeType(4.0f, juce::PathStrokeType::curved));

    g.setColour(juce::Colour(0xffa8a8b0));
    g.strokePath(curvePath, juce::PathStrokeType(2.0f, juce::PathStrokeType::curved));

    // Bright highlight line
    g.setColour(juce::Colour(0xffd0d0d8));
    g.strokePath(curvePath, juce::PathStrokeType(1.0f, juce::PathStrokeType::curved));
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

            // Chrome point with subtle glow
            g.setColour(juce::Colour(0xff404048));
            g.fillEllipse(x - 4.0f, y - 4.0f, 8.0f, 8.0f);
            g.setColour(juce::Colour(0xffc0c0c8));
            g.fillEllipse(x - 3.0f, y - 3.0f, 6.0f, 6.0f);
            g.setColour(juce::Colour(0xffe8e8f0));
            g.fillEllipse(x - 1.5f, y - 2.0f, 3.0f, 3.0f);
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
        g.setColour(juce::Colour(0xff606068));
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

        g.setColour(juce::Colour(0xffc0c0c8).withAlpha(0.15f));
        g.fillRoundedRectangle(graphArea.toFloat(), 5.0f);

        g.setColour(juce::Colour(0xffc0c0c8));
        g.setFont(16.0f);
        g.drawText("Drop CSV file here", getLocalBounds(), juce::Justification::centred);
    }

    // Drawing indicator
    if (isDrawing)
    {
        g.setColour(juce::Colour(0xffc0c0c8));
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
// Look and Feel Implementation - Liquid Metal Theme
//==============================================================================
PhaseCorrectorLookAndFeel::PhaseCorrectorLookAndFeel()
{
    // Liquid metal / chrome accent color
    accentColour = juce::Colour(0xffa0a0a8);  // Silver/chrome
    backgroundColour = juce::Colour(0xff0a0a0f);

    setColour(juce::Slider::rotarySliderFillColourId, accentColour);
    setColour(juce::Slider::rotarySliderOutlineColourId, juce::Colour(0xff252530));
    setColour(juce::ComboBox::backgroundColourId, juce::Colour(0xff101015));
    setColour(juce::ComboBox::textColourId, juce::Colour(0xffc0c0c0));
    setColour(juce::ComboBox::outlineColourId, juce::Colour(0xff353540));
    setColour(juce::ComboBox::arrowColourId, accentColour);
    setColour(juce::PopupMenu::backgroundColourId, juce::Colour(0xff101015));
    setColour(juce::PopupMenu::textColourId, juce::Colour(0xffc0c0c0));
    setColour(juce::PopupMenu::highlightedBackgroundColourId, juce::Colour(0xff303040));
    setColour(juce::Label::textColourId, juce::Colour(0xffc0c0c0));
}

void PhaseCorrectorLookAndFeel::drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
                                                  float sliderPos, float rotaryStartAngle, float rotaryEndAngle,
                                                  juce::Slider& slider)
{
    const float radius = static_cast<float>(juce::jmin(width, height)) / 2.0f - 4.0f;
    const float centreX = static_cast<float>(x) + static_cast<float>(width) * 0.5f;
    const float centreY = static_cast<float>(y) + static_cast<float>(height) * 0.5f;
    const float angle = rotaryStartAngle + sliderPos * (rotaryEndAngle - rotaryStartAngle);

    // Background arc - dark recessed look
    juce::Path backgroundArc;
    backgroundArc.addCentredArc(centreX, centreY, radius, radius, 0.0f,
                                 rotaryStartAngle, rotaryEndAngle, true);
    g.setColour(juce::Colour(0xff1a1a20));
    g.strokePath(backgroundArc, juce::PathStrokeType(5.0f, juce::PathStrokeType::curved,
                                                      juce::PathStrokeType::rounded));
    g.setColour(juce::Colour(0xff252530));
    g.strokePath(backgroundArc, juce::PathStrokeType(4.0f, juce::PathStrokeType::curved,
                                                      juce::PathStrokeType::rounded));

    // Value arc - liquid metal gradient effect
    juce::Path valueArc;
    valueArc.addCentredArc(centreX, centreY, radius, radius, 0.0f,
                           rotaryStartAngle, angle, true);

    // Chrome/silver gradient on the value arc
    g.setColour(juce::Colour(0xff606068));
    g.strokePath(valueArc, juce::PathStrokeType(5.0f, juce::PathStrokeType::curved,
                                                 juce::PathStrokeType::rounded));
    g.setColour(juce::Colour(0xffa0a0a8));
    g.strokePath(valueArc, juce::PathStrokeType(3.0f, juce::PathStrokeType::curved,
                                                 juce::PathStrokeType::rounded));
    g.setColour(juce::Colour(0xffd0d0d8));
    g.strokePath(valueArc, juce::PathStrokeType(1.5f, juce::PathStrokeType::curved,
                                                 juce::PathStrokeType::rounded));

    // Pointer - chrome needle
    juce::Path pointer;
    const float pointerLength = radius * 0.6f;
    const float pointerThickness = 3.0f;
    pointer.addRoundedRectangle(-pointerThickness * 0.5f, -radius + 4.0f,
                                 pointerThickness, pointerLength, 1.0f);
    g.setColour(juce::Colour(0xffe0e0e8));
    g.fillPath(pointer, juce::AffineTransform::rotation(angle).translated(centreX, centreY));

    // Center knob - metallic look
    juce::ColourGradient knobGradient(juce::Colour(0xff303038), centreX - 6.0f, centreY - 6.0f,
                                       juce::Colour(0xff181820), centreX + 6.0f, centreY + 6.0f, true);
    g.setGradientFill(knobGradient);
    g.fillEllipse(centreX - 8.0f, centreY - 8.0f, 16.0f, 16.0f);
    g.setColour(juce::Colour(0xff404048));
    g.drawEllipse(centreX - 8.0f, centreY - 8.0f, 16.0f, 16.0f, 1.0f);

    // Value text
    g.setColour(juce::Colour(0xffa0a0a8));
    g.setFont(std::max(9.0f, static_cast<float>(width) / 7.0f));
    juce::String valueText = slider.getTextFromValue(slider.getValue());
    g.drawText(valueText, x, static_cast<int>(centreY + radius * 0.5f),
               width, 15, juce::Justification::centred, false);
}

void PhaseCorrectorLookAndFeel::drawComboBox(juce::Graphics& g, int width, int height, bool,
                                              int, int, int, int, juce::ComboBox& box)
{
    // Metallic combo box background
    juce::ColourGradient bgGradient(juce::Colour(0xff181820), 0.0f, 0.0f,
                                     juce::Colour(0xff101015), 0.0f, static_cast<float>(height), false);
    g.setGradientFill(bgGradient);
    g.fillRoundedRectangle(0, 0, static_cast<float>(width), static_cast<float>(height), 4.0f);

    // Subtle inner shadow at top
    g.setColour(juce::Colour(0x20000000));
    g.drawHorizontalLine(1, 2.0f, static_cast<float>(width) - 2.0f);

    // Border
    g.setColour(juce::Colour(0xff353540));
    g.drawRoundedRectangle(0.5f, 0.5f, static_cast<float>(width) - 1.0f,
                            static_cast<float>(height) - 1.0f, 4.0f, 1.0f);

    // Chrome arrow
    juce::Rectangle<int> arrowZone(width - 25, 0, 20, height);
    juce::Path path;
    path.startNewSubPath(static_cast<float>(arrowZone.getX()) + 3.0f,
                         static_cast<float>(arrowZone.getCentreY()) - 2.0f);
    path.lineTo(static_cast<float>(arrowZone.getCentreX()),
                static_cast<float>(arrowZone.getCentreY()) + 3.0f);
    path.lineTo(static_cast<float>(arrowZone.getRight()) - 3.0f,
                static_cast<float>(arrowZone.getCentreY()) - 2.0f);

    g.setColour(juce::Colour(0xff808088));
    g.strokePath(path, juce::PathStrokeType(2.0f));
}

void PhaseCorrectorLookAndFeel::drawToggleButton(juce::Graphics& g, juce::ToggleButton& button,
                                                  bool shouldDrawButtonAsHighlighted, bool)
{
    auto bounds = button.getLocalBounds().toFloat().reduced(2.0f);

    // Metallic button background
    if (button.getToggleState())
    {
        juce::ColourGradient activeGradient(juce::Colour(0xff404048), bounds.getX(), bounds.getY(),
                                             juce::Colour(0xff252530), bounds.getX(), bounds.getBottom(), false);
        g.setGradientFill(activeGradient);
    }
    else
    {
        g.setColour(juce::Colour(0xff101015));
    }
    g.fillRoundedRectangle(bounds, 4.0f);

    // Border
    g.setColour(button.getToggleState() ? juce::Colour(0xff606068) : juce::Colour(0xff353540));
    g.drawRoundedRectangle(bounds, 4.0f, shouldDrawButtonAsHighlighted ? 1.5f : 1.0f);

    // Text
    g.setColour(button.getToggleState() ? juce::Colour(0xffd0d0d8) : juce::Colour(0xff707078));
    g.setFont(std::max(10.0f, bounds.getHeight() * 0.5f));
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

    // Enable resizing
    setResizable(true, true);
    setResizeLimits(MIN_WIDTH, MIN_HEIGHT, MAX_WIDTH, MAX_HEIGHT);
    getConstrainer()->setFixedAspectRatio(static_cast<double>(BASE_WIDTH) / BASE_HEIGHT);

    // Title
    titleLabel.setText("PhaseCorrector", juce::dontSendNotification);
    titleLabel.setFont(juce::FontOptions(24.0f, juce::Font::bold));
    titleLabel.setColour(juce::Label::textColourId, juce::Colour(0xffc0c0c0));  // Silver text
    titleLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(titleLabel);

    // Status label
    statusLabel.setText("Low-THD Freeform Phase EQ | 64-bit | 64x Overlap", juce::dontSendNotification);
    statusLabel.setFont(juce::FontOptions(11.0f));
    statusLabel.setColour(juce::Label::textColourId, juce::Colour(0xff808080));
    statusLabel.setJustificationType(juce::Justification::centredRight);
    addAndMakeVisible(statusLabel);

    // Phase curve display
    addAndMakeVisible(curveDisplay);

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
            curveDisplay.loadFromProcessor();
        }
    };
    addAndMakeVisible(presetBox);
    refreshPresetList();

    savePresetButton.setButtonText("Save");
    savePresetButton.onClick = [this]()
    {
        auto* alertWindow = new juce::AlertWindow("Save Preset", "Enter preset name:", juce::MessageBoxIconType::NoIcon);
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
    // Liquid metal background - subtle, premium feel
    float w = static_cast<float>(getWidth());
    float h = static_cast<float>(getHeight());

    // Base gradient - deep charcoal to dark steel
    juce::ColourGradient baseGradient(
        juce::Colour(0xff0a0a0f), 0.0f, 0.0f,
        juce::Colour(0xff141418), 0.0f, h, false);
    g.setGradientFill(baseGradient);
    g.fillAll();

    // Subtle liquid metal reflections - very understated
    // Top highlight (simulates light hitting polished surface)
    juce::ColourGradient topReflection(
        juce::Colour(0x15ffffff), 0.0f, 0.0f,
        juce::Colour(0x00ffffff), 0.0f, h * 0.3f, false);
    g.setGradientFill(topReflection);
    g.fillRect(0.0f, 0.0f, w, h * 0.3f);

    // Subtle diagonal highlight (brushed metal effect)
    juce::ColourGradient diagonalSheen(
        juce::Colour(0x00000000), 0.0f, h * 0.3f,
        juce::Colour(0x08c0c0c0), w * 0.5f, h * 0.5f, false);
    diagonalSheen.addColour(1.0, juce::Colour(0x00000000));
    g.setGradientFill(diagonalSheen);
    g.fillAll();

    // Very subtle edge highlight along top
    g.setColour(juce::Colour(0x20606060));
    g.drawHorizontalLine(0, 0.0f, w);
    g.setColour(juce::Colour(0x10404040));
    g.drawHorizontalLine(1, 0.0f, w);

    // Control panel area - slightly darker, recessed look
    float scale = static_cast<float>(getWidth()) / BASE_WIDTH;
    int controlsY = static_cast<int>(360 * scale);

    // Recessed panel effect
    juce::ColourGradient panelGradient(
        juce::Colour(0xff080808), 0.0f, static_cast<float>(controlsY),
        juce::Colour(0xff0c0c10), 0.0f, h, false);
    g.setGradientFill(panelGradient);
    g.fillRect(0, controlsY, getWidth(), getHeight() - controlsY);

    // Top edge of control panel - subtle bevel
    g.setColour(juce::Colour(0x30000000));
    g.drawHorizontalLine(controlsY, 0.0f, w);
    g.setColour(juce::Colour(0x15303035));
    g.drawHorizontalLine(controlsY + 1, 0.0f, w);

    // Bottom edge highlight
    g.setColour(juce::Colour(0x08ffffff));
    g.drawHorizontalLine(getHeight() - 1, 0.0f, w);
}

void PhaseCorrectorAudioProcessorEditor::resized()
{
    float scale = static_cast<float>(getWidth()) / BASE_WIDTH;

    // Header
    int headerH = static_cast<int>(45 * scale);
    titleLabel.setBounds(static_cast<int>(15 * scale), 0,
                         static_cast<int>(180 * scale), headerH);

    // Clear and Invert buttons in header
    clearButton.setBounds(static_cast<int>(195 * scale), static_cast<int>(10 * scale),
                          static_cast<int>(50 * scale), static_cast<int>(25 * scale));
    invertButton.setBounds(static_cast<int>(250 * scale), static_cast<int>(10 * scale),
                           static_cast<int>(50 * scale), static_cast<int>(25 * scale));

    // Preset controls in header
    presetBox.setBounds(static_cast<int>(315 * scale), static_cast<int>(10 * scale),
                        static_cast<int>(130 * scale), static_cast<int>(25 * scale));
    savePresetButton.setBounds(static_cast<int>(450 * scale), static_cast<int>(10 * scale),
                               static_cast<int>(40 * scale), static_cast<int>(25 * scale));
    deletePresetButton.setBounds(static_cast<int>(495 * scale), static_cast<int>(10 * scale),
                                 static_cast<int>(35 * scale), static_cast<int>(25 * scale));

    statusLabel.setBounds(getWidth() - static_cast<int>(250 * scale), 0,
                          static_cast<int>(240 * scale), headerH);

    // Phase curve display (main area - larger now without nyquist visualizer)
    int curveY = headerH;
    int curveH = static_cast<int>(305 * scale);
    curveDisplay.setBounds(static_cast<int>(10 * scale), curveY,
                           getWidth() - static_cast<int>(20 * scale), curveH);

    // Controls area (directly below curve)
    int controlsY = curveY + curveH + static_cast<int>(10 * scale);
    int knobSize = static_cast<int>(70 * scale);
    int labelH = static_cast<int>(15 * scale);
    int spacing = static_cast<int>(20 * scale);

    // Center the controls
    int totalControlsWidth = static_cast<int>(100 * scale) * 2 + static_cast<int>(10 * scale) +
                              knobSize * 3 + spacing * 2;
    int startX = (getWidth() - totalControlsWidth) / 2;

    int x = startX;
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
