# Import necessary libraries
from flask import Flask, request, jsonify, render_template_string
import random 

# Create a Flask application instance
app = Flask(__name__)

# --- Model and Data Simulation ---
# In a full project, you would load your H5 model here.
# For this demonstration, we'll simulate the model's output.
# To load a real .H5 model, you would use:
# from tensorflow.keras.models import load_model
# model = load_model('croprecommender_mlp.H5')

# Unique lists provided by the user and additional data
UNIQUE_PRIMARY_CROPS = ['Paddy', 'Bengal Gram', 'Vegetables', 'Cotton', 'Chillies', 'Maize', 'Groundnut', 'Pearl Millet']
UNIQUE_WATER_SOURCES = ['Tank', 'Borewell', 'Canal']
SEASONS = ['Kharif', 'Rabi', 'Zaid', 'Whole Year']
SOIL_TYPES = ['Clay', 'Loam', 'Sandy']

# Simulated data for auto-populating soil and season based on district.
# This data also includes new, simulated "real-world" properties.
DISTRICT_PROPERTIES = {
    'Anantapur': {'soil_type': 'Sandy', 'season': 'Rabi', 'rainfall_mm': 544, 'common_crops': ['Groundnut', 'Pearl Millet']},
    'Chittoor': {'soil_type': 'Loam', 'season': 'Kharif', 'rainfall_mm': 876, 'common_crops': ['Groundnut', 'Paddy']},
    'East Godavari': {'soil_type': 'Clay', 'season': 'Whole Year', 'rainfall_mm': 1145, 'common_crops': ['Paddy', 'Maize']},
    'Guntur': {'soil_type': 'Clay', 'season': 'Kharif', 'rainfall_mm': 920, 'common_crops': ['Chillies', 'Paddy']},
    'Kadapa': {'soil_type': 'Sandy', 'season': 'Rabi', 'rainfall_mm': 694, 'common_crops': ['Paddy', 'Bengal Gram']},
    'Krishna': {'soil_type': 'Clay', 'season': 'Whole Year', 'rainfall_mm': 978, 'common_crops': ['Paddy', 'Vegetables']},
    'Kurnool': {'soil_type': 'Loam', 'season': 'Rabi', 'rainfall_mm': 614, 'common_crops': ['Cotton', 'Paddy']},
    'Nellore': {'soil_type': 'Clay', 'season': 'Kharif', 'rainfall_mm': 1083, 'common_crops': ['Paddy', 'Chillies']},
    'Prakasam': {'soil_type': 'Sandy', 'season': 'Kharif', 'rainfall_mm': 847, 'common_crops': ['Cotton', 'Groundnut']},
    'Srikakulam': {'soil_type': 'Clay', 'season': 'Whole Year', 'rainfall_mm': 1109, 'common_crops': ['Paddy', 'Pearl Millet']},
    'Visakhapatnam': {'soil_type': 'Loam', 'season': 'Kharif', 'rainfall_mm': 1120, 'common_crops': ['Paddy', 'Maize']},
    'Vizianagaram': {'soil_type': 'Loam', 'season': 'Kharif', 'rainfall_mm': 1060, 'common_crops': ['Paddy', 'Vegetables']},
    'West Godavari': {'soil_type': 'Clay', 'season': 'Whole Year', 'rainfall_mm': 1080, 'common_crops': ['Paddy', 'Cotton']}
}

# --- Simulated Data for Guidance Features ---
# This data simulates the kind of information that would come from your datasets,
# with added properties like crop duration and pH.
IRRIGATION_DATA = {
    'Paddy': {
        'methods': {'Borewell': 'Drip/Sprinkler', 'Tank': 'Flood', 'Canal': 'Flood'},
        'schedule': {'Borewell': '10 irrigations', 'Tank': '5-6 irrigations', 'Canal': 'Continuous flow'},
        'water_saving': 'Alternate wetting and drying (AWD)'
    },
    'Groundnut': {
        'methods': {'Borewell': 'Drip/Sprinkler', 'Tank': 'Sprinkler', 'Canal': 'Flood'},
        'schedule': {'Borewell': '5-7 irrigations', 'Tank': '2-3 irrigations', 'Canal': '2-3 irrigations'},
        'water_saving': 'Use of mulch to retain moisture'
    },
    'Cotton': {
        'methods': {'Borewell': 'Drip', 'Tank': 'Furrow', 'Canal': 'Furrow'},
        'schedule': {'Borewell': '4-5 irrigations', 'Tank': '3-4 irrigations', 'Canal': '3-4 irrigations'},
        'water_saving': 'Precision irrigation timing with soil moisture sensors'
    },
    # Add more crop data here
}

FERTILIZER_DATA = {
    'Paddy': {'N_rec': '100-120 kg/ha', 'P_rec': '40-50 kg/ha', 'K_rec': '40-50 kg/ha', 'organic': '2-3 tons of farmyard manure'},
    'Groundnut': {'N_rec': '20-30 kg/ha', 'P_rec': '40-50 kg/ha', 'K_rec': '40-50 kg/ha', 'organic': '1-2 tons of compost'},
    'Cotton': {'N_rec': '80-100 kg/ha', 'P_rec': '40-50 kg/ha', 'K_rec': '40-50 kg/ha', 'organic': '2-3 tons of vermicompost'},
    # Add more crop data here
}

CROP_PROPERTIES = {
    'Paddy': {'duration_days': '120-140', 'optimal_ph': '6.0-7.0', 'temp_range': '20-35 °C'},
    'Groundnut': {'duration_days': '100-110', 'optimal_ph': '6.0-6.5', 'temp_range': '25-30 °C'},
    'Cotton': {'duration_days': '150-180', 'optimal_ph': '6.0-7.5', 'temp_range': '21-30 °C'},
}

MARKET_DATA = {
    'Anantapur': {
        'Paddy': {'yield_trend': 'Stable', 'price_trend': 'Moderate increase', 'risk': 'Low'},
        'Groundnut': {'yield_trend': 'Slightly decreasing due to variable rainfall', 'price_trend': 'High and stable', 'risk': 'Medium'},
    },
    'Guntur': {
        'Paddy': {'yield_trend': 'Increasing', 'price_trend': 'Stable', 'risk': 'Very low'},
        'Chillies': {'yield_trend': 'Volatile', 'price_trend': 'High but fluctuates', 'risk': 'High'},
    },
    # Add more district data here
}


# --- Functions to Simulate Logic ---

def get_crop_recommendations(user_input):
    """
    Simulates the ML model to provide a ranked list of crop recommendations.
    This is where your H5 model's prediction logic would go.
    """
    recommendations = []
    
    # Use a simple, hardcoded rule based on input for a more 'intelligent' simulation
    # In a real model, this would be a single `model.predict()` call
    if user_input.get('state') == 'Andhra Pradesh':
        # Condition for Paddy
        if float(user_input.get('n-ppm', 0)) > 80 and user_input.get('water-resource') == 'Canal':
            recommendations.append({'crop': 'Paddy', 'score': random.uniform(0.9, 0.98)})
        # Condition for Groundnut
        if float(user_input.get('p-ppm', 0)) > 40 and user_input.get('soil-type') == 'Sandy':
            recommendations.append({'crop': 'Groundnut', 'score': random.uniform(0.85, 0.95)})
        # Condition for Chillies
        if float(user_input.get('k-ppm', 0)) > 50 and user_input.get('avg-temp', 0) > 25:
            recommendations.append({'crop': 'Chillies', 'score': random.uniform(0.8, 0.9)})

    # Fill up the rest with random crops from the list
    other_crops = [crop for crop in UNIQUE_PRIMARY_CROPS if crop not in [rec['crop'] for rec in recommendations]]
    random.shuffle(other_crops)
    
    while len(recommendations) < 3 and other_crops:
        crop = other_crops.pop(0)
        score = random.uniform(0.65, 0.8)
        recommendations.append({'crop': crop, 'score': score})
    
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    return recommendations[:3]


def get_location_details(lat, lon):
    """
    Simulates auto-detecting district and mandal from lat/long.
    This is a placeholder for a real geocoding service or dataset lookup.
    """
    # A very simple, fixed simulation
    if lat and lon:
        return {'district': 'Anantapur', 'mandal': 'Kadiri'}
    return {'district': None, 'mandal': None}


def get_irrigation_guidance(crop, soil_type, water_source):
    """
    Provides irrigation recommendations based on simulated data.
    """
    guidance = IRRIGATION_DATA.get(crop, {})
    return {
        'method': guidance.get('methods', {}).get(water_source, 'Drip irrigation recommended'),
        'schedule': guidance.get('schedule', {}).get(water_source, 'Consult local agricultural officer'),
        'water_saving_practices': guidance.get('water_saving', 'Use drought-resistant varieties')
    }


def get_fertilizer_plan(crop, n, p, k):
    """
    Provides a fertilizer plan based on simulated crop and soil data.
    """
    plan = FERTILIZER_DATA.get(crop, {})
    return {
        'npk_doses': plan.get('N_rec', 'N/A') + ', ' + plan.get('P_rec', 'N/A') + ', ' + plan.get('K_rec', 'N/A'),
        'organic_manure': plan.get('organic', 'Add compost or farmyard manure based on soil test.')
    }


def get_market_intelligence(crop, district):
    """
    Provides simulated market and seasonal insights.
    """
    insights = MARKET_DATA.get(district, {}).get(crop, {})
    return {
        'yield_patterns': insights.get('yield_trend', 'Historical yield data is unavailable.'),
        'price_trends': insights.get('price_trend', 'Market price trends are currently stable.'),
        'risk_level': insights.get('risk_level', 'Low')
    }


# --- Flask Routes ---

@app.route('/')
def home():
    """Serves the main HTML page for the web application."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AgroIntelligence - Full-Featured System</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            body { font-family: 'Inter', sans-serif; }
            ::-webkit-scrollbar { width: 8px; }
            ::-webkit-scrollbar-track { background: #f1f5f9; border-radius: 10px; }
            ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; }
            ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
        </style>
    </head>
    <body class="bg-gray-100 flex items-center justify-center min-h-screen p-4">
        <div class="bg-white p-8 rounded-3xl shadow-2xl w-full max-w-5xl transform transition-all duration-300 hover:shadow-3xl">
            <h1 class="text-4xl font-bold text-center text-slate-800 mb-6">
                <i class="fas fa-seedling text-green-500 mr-2 animate-bounce"></i> AgroIntelligence
            </h1>
            <p class="text-center text-slate-500 mb-8">
                Your one-stop farm guidance system.
            </p>

            <form id="recommendation-form" class="space-y-6">
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <!-- Location & Environmental Inputs -->
                    <div class="space-y-4">
                        <div class="col-span-full"><h3 class="text-lg font-bold text-slate-700">Location & Weather</h3></div>
                        <div>
                            <label for="state" class="block text-sm font-medium text-slate-700 mb-1">State</label>
                            <select id="state" name="state" required class="w-full px-4 py-2 border border-slate-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition duration-200">
                                <option value="" disabled selected>Select State</option>
                                <option value="Andhra Pradesh">Andhra Pradesh</option>
                            </select>
                        </div>
                        <div>
                            <label for="district" class="block text-sm font-medium text-slate-700 mb-1">District</label>
                            <select id="district" name="district" required class="w-full px-4 py-2 border border-slate-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition duration-200">
                                <option value="" disabled selected>Select District</option>
                                <!-- Districts will be populated by JS -->
                            </select>
                        </div>
                        <div>
                            <label for="season" class="block text-sm font-medium text-slate-700 mb-1">Season</label>
                            <select id="season" name="season" required class="w-full px-4 py-2 border border-slate-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition duration-200">
                                <option value="" disabled selected>Select Season</option>
                                <option value="Kharif">Kharif</option>
                                <option value="Rabi">Rabi</option>
                                <option value="Zaid">Zaid</option>
                                <option value="Whole Year">Whole Year</option>
                            </select>
                        </div>
                        <div>
                            <label for="avg-temp" class="block text-sm font-medium text-slate-700 mb-1">Average Temperature (°C)</label>
                            <input type="number" step="0.1" id="avg-temp" name="avg-temp" required class="w-full px-4 py-2 border border-slate-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition duration-200" placeholder="e.g., 20.87">
                        </div>
                        <div>
                            <label for="avg-humidity" class="block text-sm font-medium text-slate-700 mb-1">Average Humidity (%)</label>
                            <input type="number" step="0.1" id="avg-humidity" name="avg-humidity" required class="w-full px-4 py-2 border border-slate-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition duration-200" placeholder="e.g., 82.00">
                        </div>
                    </div>
                    
                    <!-- Soil & Crop Inputs -->
                    <div class="space-y-4">
                        <div class="col-span-full"><h3 class="text-lg font-bold text-slate-700">Soil Details</h3></div>
                        <div>
                            <label for="soil-type" class="block text-sm font-medium text-slate-700 mb-1">Soil Type</label>
                            <select id="soil-type" name="soil-type" required class="w-full px-4 py-2 border border-slate-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition duration-200">
                                <option value="" disabled selected>Select Soil Type</option>
                                <option value="Clay">Clay</option>
                                <option value="Loam">Loam</option>
                                <option value="Sandy">Sandy</option>
                            </select>
                        </div>
                        <div>
                            <label for="water-resource" class="block text-sm font-medium text-slate-700 mb-1">Water Resource</label>
                            <select id="water-resource" name="water-resource" required class="w-full px-4 py-2 border border-slate-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition duration-200">
                                <option value="" disabled selected>Select Water Resource</option>
                                <option value="Tank">Tank</option>
                                <option value="Borewell">Borewell</option>
                                <option value="Canal">Canal</option>
                            </select>
                        </div>
                        <div>
                            <label for="n-ppm" class="block text-sm font-medium text-slate-700 mb-1">Nitrogen (N) in ppm</label>
                            <input type="number" id="n-ppm" name="n-ppm" required class="w-full px-4 py-2 border border-slate-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition duration-200" placeholder="e.g., 90">
                        </div>
                        <div>
                            <label for="p-ppm" class="block text-sm font-medium text-slate-700 mb-1">Phosphorus (P) in ppm</label>
                            <input type="number" id="p-ppm" name="p-ppm" required class="w-full px-4 py-2 border border-slate-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition duration-200" placeholder="e.g., 42">
                        </div>
                        <div>
                            <label for="k-ppm" class="block text-sm font-medium text-slate-700 mb-1">Potassium (K) in ppm</label>
                            <input type="number" id="k-ppm" name="k-ppm" required class="w-full px-4 py-2 border border-slate-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition duration-200" placeholder="e.g., 43">
                        </div>
                    </div>

                    <!-- Geo-location Inputs (optional) -->
                    <div class="space-y-4">
                        <div class="col-span-full"><h3 class="text-lg font-bold text-slate-700">Optional: Auto-Detect Location</h3></div>
                        <div>
                            <label for="latitude" class="block text-sm font-medium text-slate-700 mb-1">Latitude</label>
                            <input type="number" step="0.000001" id="latitude" name="latitude" class="w-full px-4 py-2 border border-slate-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition duration-200" placeholder="e.g., 14.6819">
                        </div>
                        <div>
                            <label for="longitude" class="block text-sm font-medium text-slate-700 mb-1">Longitude</label>
                            <input type="number" step="0.000001" id="longitude" name="longitude" class="w-full px-4 py-2 border border-slate-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition duration-200" placeholder="e.g., 77.6006">
                        </div>
                    </div>
                </div>

                <div class="pt-6 text-center">
                    <button type="submit" class="w-full md:w-1/2 bg-green-600 text-white font-semibold py-3 px-6 rounded-lg shadow-lg hover:bg-green-700 focus:outline-none focus:ring-4 focus:ring-green-300 transition duration-300 transform hover:scale-105">
                        Get Full Farm Guidance
                    </button>
                </div>
            </form>

            <div id="loading" class="mt-8 text-center hidden">
                <i class="fas fa-spinner fa-spin text-4xl text-green-500"></i>
                <p class="mt-2 text-slate-600">Calculating recommendations and farm plan...</p>
            </div>

            <div id="results" class="mt-8 hidden space-y-8">
                <!-- Location Details -->
                <div id="location-details" class="p-6 bg-slate-50 rounded-2xl shadow-inner hidden">
                    <h2 class="text-2xl font-bold text-slate-800 mb-2">Location Details</h2>
                    <p class="text-slate-600"><span class="font-semibold">District:</span> <span id="output-district"></span></p>
                    <p class="text-slate-600"><span class="font-semibold">Mandal:</span> <span id="output-mandal"></span></p>
                    <p class="text-slate-600"><span class="font-semibold">Annual Rainfall:</span> <span id="output-rainfall"></span> mm</p>
                </div>

                <!-- Recommendations -->
                <div>
                    <h2 class="text-2xl font-bold text-center text-slate-800 mb-4">Top Crop Recommendations</h2>
                    <div id="recommendation-list" class="space-y-4"></div>
                </div>

                <!-- Guidance for the Top-Recommended Crop -->
                <div id="guidance-section" class="p-6 bg-green-50 rounded-2xl shadow-lg hidden">
                    <h2 class="text-2xl font-bold text-green-800 mb-4 text-center">Guidance for: <span id="primary-crop" class="text-green-600"></span></h2>
                    
                    <!-- Crop Properties -->
                    <div class="mb-6">
                        <h3 class="text-xl font-bold text-green-700 mb-2"><i class="fas fa-info-circle mr-2"></i>Crop Details</h3>
                        <div class="bg-white p-4 rounded-xl shadow-md border border-green-200 space-y-2">
                            <p class="text-slate-700"><span class="font-semibold">Crop Duration:</span> <span id="crop-duration"></span></p>
                            <p class="text-slate-700"><span class="font-semibold">Optimal pH:</span> <span id="optimal-ph"></span></p>
                            <p class="text-slate-700"><span class="font-semibold">Temperature Range:</span> <span id="temp-range"></span></p>
                        </div>
                    </div>

                    <!-- Irrigation Guidance -->
                    <div class="mb-6">
                        <h3 class="text-xl font-bold text-green-700 mb-2"><i class="fas fa-water mr-2"></i>Irrigation Guidance</h3>
                        <div class="bg-white p-4 rounded-xl shadow-md border border-green-200 space-y-2">
                            <p class="text-slate-700"><span class="font-semibold">Method:</span> <span id="irrigation-method"></span></p>
                            <p class="text-slate-700"><span class="font-semibold">Schedule:</span> <span id="irrigation-schedule"></span></p>
                            <p class="text-slate-700"><span class="font-semibold">Water-Saving:</span> <span id="irrigation-saving"></span></p>
                        </div>
                    </div>

                    <!-- Fertilizer Plan -->
                    <div class="mb-6">
                        <h3 class="text-xl font-bold text-green-700 mb-2"><i class="fas fa-flask mr-2"></i>Fertilizer Plan</h3>
                        <div class="bg-white p-4 rounded-xl shadow-md border border-green-200 space-y-2">
                            <p class="text-slate-700"><span class="font-semibold">NPK Doses:</span> <span id="fertilizer-npk"></span></p>
                            <p class="text-slate-700"><span class="font-semibold">Organic Manure:</span> <span id="fertilizer-organic"></span></p>
                        </div>
                    </div>

                    <!-- Market Intelligence -->
                    <div>
                        <h3 class="text-xl font-bold text-green-700 mb-2"><i class="fas fa-chart-line mr-2"></i>Market Intelligence</h3>
                        <div class="bg-white p-4 rounded-xl shadow-md border border-green-200 space-y-2">
                            <p class="text-slate-700"><span class="font-semibold">Yield Patterns:</span> <span id="market-yield"></span></p>
                            <p class="text-slate-700"><span class="font-semibold">Price Trends:</span> <span id="market-price"></span></p>
                            <p class="text-slate-700"><span class="font-semibold">Risk Level:</span> <span id="market-risk"></span></p>
                        </div>
                    </div>
                </div>

            </div>

            <div id="error-message" class="mt-8 hidden p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg" role="alert">
                <p class="font-bold">An error occurred!</p>
                <p id="error-text" class="text-sm mt-1"></p>
            </div>
        </div>

        <script>
            // Populate the district dropdown dynamically
            async function populateDistricts() {
                const response = await fetch('/get_district_names');
                const districts = await response.json();
                const districtSelect = document.getElementById('district');
                districts.forEach(district => {
                    const option = document.createElement('option');
                    option.value = district;
                    option.textContent = district;
                    districtSelect.appendChild(option);
                });
            }
            populateDistricts();

            // Handle auto-population based on district selection
            document.getElementById('district').addEventListener('change', async function(event) {
                const selectedDistrict = event.target.value;
                const response = await fetch('/get_district_data/' + selectedDistrict);
                const data = await response.json();
                if (data) {
                    document.getElementById('soil-type').value = data.soil_type;
                    document.getElementById('season').value = data.season;
                }
            });

            document.getElementById('recommendation-form').addEventListener('submit', async function(event) {
                event.preventDefault();
                const form = document.getElementById('recommendation-form');
                const loadingIndicator = document.getElementById('loading');
                const resultsDiv = document.getElementById('results');
                const recommendationList = document.getElementById('recommendation-list');
                const guidanceSection = document.getElementById('guidance-section');
                const locationDetails = document.getElementById('location-details');
                const errorMessageDiv = document.getElementById('error-message');
                const errorText = document.getElementById('error-text');

                // Reset outputs
                resultsDiv.classList.add('hidden');
                guidanceSection.classList.add('hidden');
                locationDetails.classList.add('hidden');
                errorMessageDiv.classList.add('hidden');
                loadingIndicator.classList.remove('hidden');

                const formData = new FormData(form);
                const data = Object.fromEntries(formData.entries());

                // Parse numeric values
                data['n-ppm'] = parseFloat(data['n-ppm']);
                data['p-ppm'] = parseFloat(data['p-ppm']);
                data['k-ppm'] = parseFloat(data['k-ppm']);
                data['avg-temp'] = parseFloat(data['avg-temp']);
                data['avg-humidity'] = parseFloat(data['avg-humidity']);
                data['latitude'] = parseFloat(data['latitude']) || null;
                data['longitude'] = parseFloat(data['longitude']) || null;

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data),
                    });

                    const result = await response.json();

                    if (response.ok) {
                        loadingIndicator.classList.add('hidden');
                        resultsDiv.classList.remove('hidden');
                        
                        // Render Location Details
                        if (result.location_details.district) {
                            document.getElementById('output-district').textContent = result.location_details.district;
                            document.getElementById('output-mandal').textContent = result.location_details.mandal;
                            // Add new rainfall data
                            const districtDataResponse = await fetch('/get_district_data/' + result.location_details.district);
                            const districtData = await districtDataResponse.json();
                            document.getElementById('output-rainfall').textContent = districtData.rainfall_mm;
                            locationDetails.classList.remove('hidden');
                        }

                        // Render Crop Recommendations
                        recommendationList.innerHTML = '';
                        result.recommendations.forEach((rec, index) => {
                            const score = (rec.score * 100).toFixed(2);
                            const itemHtml = `
                                <div class="flex items-center p-4 bg-gray-50 rounded-lg shadow-md border-l-4 border-green-500">
                                    <span class="text-xl font-bold text-green-600 mr-4">${index + 1}.</span>
                                    <div class="flex-grow">
                                        <p class="text-lg font-semibold text-slate-800">${rec.crop}</p>
                                    </div>
                                    <span class="bg-green-200 text-green-800 text-sm font-semibold px-3 py-1 rounded-full">
                                        ${score}% Suitability
                                    </span>
                                </div>
                            `;
                            recommendationList.innerHTML += itemHtml;
                        });

                        // Render Guidance for the top crop
                        const topCrop = result.recommendations[0].crop;
                        document.getElementById('primary-crop').textContent = topCrop;
                        
                        // Add new crop properties
                        const cropPropertiesResponse = await fetch('/get_crop_data/' + topCrop);
                        const cropProperties = await cropPropertiesResponse.json();
                        document.getElementById('crop-duration').textContent = cropProperties.duration_days;
                        document.getElementById('optimal-ph').textContent = cropProperties.optimal_ph;
                        document.getElementById('temp-range').textContent = cropProperties.temp_range;

                        document.getElementById('irrigation-method').textContent = result.irrigation_guidance.method;
                        document.getElementById('irrigation-schedule').textContent = result.irrigation_guidance.schedule;
                        document.getElementById('irrigation-saving').textContent = result.irrigation_guidance.water_saving_practices;
                        
                        document.getElementById('fertilizer-npk').textContent = result.fertilizer_plan.npk_doses;
                        document.getElementById('fertilizer-organic').textContent = result.fertilizer_plan.organic_manure;

                        document.getElementById('market-yield').textContent = result.market_intelligence.yield_patterns;
                        document.getElementById('market-price').textContent = result.market_intelligence.price_trends;
                        document.getElementById('market-risk').textContent = result.market_intelligence.risk_level;
                        
                        guidanceSection.classList.remove('hidden');

                    } else {
                        loadingIndicator.classList.add('hidden');
                        errorMessageDiv.classList.remove('hidden');
                        errorText.textContent = result.error || 'Server returned an error.';
                    }
                } catch (error) {
                    console.error('Error:', error);
                    loadingIndicator.classList.add('hidden');
                    errorMessageDiv.classList.remove('hidden');
                    errorText.textContent = 'Could not connect to the server. Please try again later.';
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

@app.route('/get_district_names')
def get_district_names():
    """Endpoint to get a list of all district names."""
    return jsonify(list(DISTRICT_PROPERTIES.keys()))

@app.route('/get_district_data/<district_name>')
def get_district_data(district_name):
    """Endpoint to get properties for a specific district."""
    return jsonify(DISTRICT_PROPERTIES.get(district_name, {}))

@app.route('/get_crop_data/<crop_name>')
def get_crop_data(crop_name):
    """Endpoint to get properties for a specific crop."""
    return jsonify(CROP_PROPERTIES.get(crop_name, {}))

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for receiving form data and returning a full farm guidance plan."""
    try:
        user_input = request.get_json()
        if not user_input:
            return jsonify({'error': 'Invalid input'}), 400
        
        # 1. Crop Recommendation
        recommendations = get_crop_recommendations(user_input)
        primary_crop = recommendations[0]['crop']

        # 2. District & Mandal Auto-Detection
        location_details = get_location_details(user_input.get('latitude'), user_input.get('longitude'))
        
        # 3. Irrigation Guidance
        irrigation_guidance = get_irrigation_guidance(primary_crop, user_input.get('soil-type'), user_input.get('water-resource'))
        
        # 4. Fertilizer Plan
        fertilizer_plan = get_fertilizer_plan(primary_crop, user_input.get('n-ppm'), user_input.get('p-ppm'), user_input.get('k-ppm'))

        # 5. Seasonal & Market Intelligence
        market_intelligence = get_market_intelligence(primary_crop, location_details.get('district', 'Anantapur'))
        
        # Combine all outputs into a single response
        response = {
            'recommendations': recommendations,
            'location_details': location_details,
            'irrigation_guidance': irrigation_guidance,
            'fertilizer_plan': fertilizer_plan,
            'market_intelligence': market_intelligence
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the application
if __name__ == '__main__':
    # You can change the port and host as needed.
    app.run(debug=True, port=5000)
