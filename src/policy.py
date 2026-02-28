def management_actions(aqi: float):
    """
    Determine AQI category and recommended management actions based on AQI value.
    
    Args:
        aqi: Predicted Air Quality Index value
        
    Returns:
        Tuple of (category, actions)
    """
    if aqi <= 50:
        category = "Good"
        actions = [
            "Air quality is satisfactory. Enjoy outdoor activities!",
            "No restrictions needed."
        ]
    elif aqi <= 100:
        category = "Moderate"
        actions = [
            "Unusually sensitive people should consider limiting prolonged outdoor exertion.",
            "No indoor air quality restrictions needed."
        ]
    elif aqi <= 150:
        category = "Unhealthy for Sensitive Groups"
        actions = [
            "Children and asthmatics should reduce outdoor exertion.",
            "Sensitive individuals should limit prolonged outdoor activities.",
            "Consider using air purifiers indoors."
        ]
    elif aqi <= 200:
        category = "Unhealthy"
        actions = [
            "Everyone may begin to experience health effects.",
            "Children and asthmatics should avoid outdoor activities.",
            "Sensitive groups should stay indoors.",
            "Wear N95 masks if going outside is necessary."
        ]
    elif aqi <= 300:
        category = "Very Unhealthy"
        actions = [
            "Health warnings of emergency conditions.",
            "Everyone is more likely to be affected.",
            "Avoid all outdoor activities.",
            "Keep windows closed and use air purifiers."
        ]
    else:
        category = "Hazardous"
        actions = [
            "Health alert: everyone may experience more serious health effects.",
            "Stay indoors with windows and doors sealed.",
            "Use emergency air filtration if available.",
            "Follow all local health advisories."
        ]
    
    return category, actions
