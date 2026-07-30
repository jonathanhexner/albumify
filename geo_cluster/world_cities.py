"""Candidate city list for StreetCLIP zero-shot geolocation (spec-022).

Not exhaustive — a spread of major/notable cities so StreetCLIP has plausible
candidates to score against when GPS is missing. Extend freely; the locator
just scores the image against every "(city, country)" prompt.
"""

WORLD_CITIES = [
    ("Budapest", "Hungary"), ("Vienna", "Austria"), ("Prague", "Czechia"),
    ("Bratislava", "Slovakia"), ("Krakow", "Poland"), ("Warsaw", "Poland"),
    ("Berlin", "Germany"), ("Munich", "Germany"), ("Hamburg", "Germany"),
    ("Paris", "France"), ("Lyon", "France"), ("Nice", "France"),
    ("London", "United Kingdom"), ("Edinburgh", "United Kingdom"),
    ("Amsterdam", "Netherlands"), ("Brussels", "Belgium"), ("Zurich", "Switzerland"),
    ("Geneva", "Switzerland"), ("Innsbruck", "Austria"), ("Salzburg", "Austria"),
    ("Rome", "Italy"), ("Milan", "Italy"), ("Venice", "Italy"), ("Florence", "Italy"),
    ("Naples", "Italy"), ("Madrid", "Spain"), ("Barcelona", "Spain"),
    ("Seville", "Spain"), ("Lisbon", "Portugal"), ("Porto", "Portugal"),
    ("Athens", "Greece"), ("Istanbul", "Turkey"), ("Dubrovnik", "Croatia"),
    ("Zagreb", "Croatia"), ("Ljubljana", "Slovenia"), ("Copenhagen", "Denmark"),
    ("Stockholm", "Sweden"), ("Oslo", "Norway"), ("Helsinki", "Finland"),
    ("Reykjavik", "Iceland"), ("Dublin", "Ireland"), ("Moscow", "Russia"),
    ("Saint Petersburg", "Russia"), ("Kyiv", "Ukraine"), ("Bucharest", "Romania"),
    ("Sofia", "Bulgaria"), ("Belgrade", "Serbia"),
    ("Tel Aviv", "Israel"), ("Jerusalem", "Israel"), ("Dubai", "United Arab Emirates"),
    ("Abu Dhabi", "United Arab Emirates"), ("Doha", "Qatar"), ("Cairo", "Egypt"),
    ("Marrakech", "Morocco"), ("Cape Town", "South Africa"), ("Nairobi", "Kenya"),
    ("New York", "United States"), ("Los Angeles", "United States"),
    ("San Francisco", "United States"), ("Chicago", "United States"),
    ("Las Vegas", "United States"), ("Miami", "United States"),
    ("Boston", "United States"), ("Seattle", "United States"),
    ("Washington", "United States"), ("Toronto", "Canada"), ("Vancouver", "Canada"),
    ("Montreal", "Canada"), ("Mexico City", "Mexico"), ("Havana", "Cuba"),
    ("Rio de Janeiro", "Brazil"), ("Sao Paulo", "Brazil"),
    ("Buenos Aires", "Argentina"), ("Lima", "Peru"), ("Santiago", "Chile"),
    ("Bogota", "Colombia"), ("Cusco", "Peru"),
    ("Tokyo", "Japan"), ("Kyoto", "Japan"), ("Osaka", "Japan"),
    ("Seoul", "South Korea"), ("Beijing", "China"), ("Shanghai", "China"),
    ("Hong Kong", "China"), ("Bangkok", "Thailand"), ("Singapore", "Singapore"),
    ("Kuala Lumpur", "Malaysia"), ("Bali", "Indonesia"), ("Jakarta", "Indonesia"),
    ("Hanoi", "Vietnam"), ("Mumbai", "India"), ("Delhi", "India"),
    ("Jaipur", "India"), ("Kathmandu", "Nepal"),
    ("Sydney", "Australia"), ("Melbourne", "Australia"), ("Auckland", "New Zealand"),
    ("Queenstown", "New Zealand"), ("Honolulu", "United States"),
]

PROMPT_TEMPLATE = "A photo taken in {city}, {country}."


def city_prompts(cities=None):
    cities = cities or WORLD_CITIES
    labels = [f"{c}, {co}" for c, co in cities]
    prompts = [PROMPT_TEMPLATE.format(city=c, country=co) for c, co in cities]
    return labels, prompts
