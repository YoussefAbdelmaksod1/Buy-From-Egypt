# Egyptian Industry Sectors with Details
INDUSTRY_SECTORS = {
    "Textiles": {
        "description": "Egypt's textile industry leverages the country's high-quality cotton production, with a focus on exports to Europe and MENA regions.",
        "recommendation_focus": "Product partnerships with clothing manufacturers, cotton processing equipment, and textile design businesses.",
        "key_regions": ["Greater Cairo", "Nile Delta", "Alexandria"],
        "seasonal_factors": ["Winter tourism increases demand for textile souvenirs", "Ramadan increases demand for home textiles"],
        "economic_sensitivity": "Medium sensitivity to tourism fluctuations and cotton price changes"
    },
    "Agriculture": {
        "description": "Agricultural businesses in Egypt focus on fruits, vegetables, and cotton, with strong export potential to Europe and Gulf countries.",
        "recommendation_focus": "Connecting with food processing businesses, agricultural equipment suppliers, and export logistics companies.",
        "key_regions": ["Nile Delta", "Upper Egypt", "Fayoum"],
        "seasonal_factors": ["Harvest seasons vary by crop", "Ramadan affects food consumption patterns"],
        "economic_sensitivity": "High sensitivity to water availability and climate conditions"
    },
    "Spices": {
        "description": "Egyptian spice producers and traders deal in traditional spices with historical significance in Middle Eastern and North African cuisine.",
        "recommendation_focus": "Partnerships with food manufacturers, exporters to tourism-heavy regions, and culinary businesses.",
        "key_regions": ["Greater Cairo", "Alexandria", "Upper Egypt"],
        "seasonal_factors": ["Tourism seasons affect demand", "Ramadan significantly increases household consumption"],
        "economic_sensitivity": "Medium sensitivity to tourism and global spice market prices"
    },
    "Fruits & Vegetables": {
        "description": "Egypt is known for citrus fruits, dates, and various vegetables with year-round growing seasons due to the climate.",
        "recommendation_focus": "Cold chain logistics, export packaging businesses, and processing facilities for value-added products.",
        "key_regions": ["Nile Delta", "Upper Egypt", "Sinai"],
        "seasonal_factors": ["Different harvest seasons throughout the year", "Export demand fluctuates with European growing seasons"],
        "economic_sensitivity": "High sensitivity to export market conditions and domestic consumption patterns"
    },
    "Chemicals": {
        "description": "The Egyptian chemicals industry produces a range of industrial chemicals, fertilizers, and petrochemicals.",
        "recommendation_focus": "Partnerships with manufacturing industries, agricultural suppliers, and construction businesses.",
        "key_regions": ["Alexandria", "Suez", "Cairo"],
        "seasonal_factors": ["Fertilizer demand varies with agricultural seasons"],
        "economic_sensitivity": "High sensitivity to energy prices and industrial production"
    },
    "Pharmaceuticals": {
        "description": "Egypt has a growing pharmaceutical industry focusing on generic medications and medical supplies for domestic and regional markets.",
        "recommendation_focus": "Medical equipment suppliers, healthcare facilities, and logistics companies with temperature-controlled capabilities.",
        "key_regions": ["Greater Cairo", "Alexandria", "10th of Ramadan City"],
        "seasonal_factors": ["Seasonal illness patterns affect demand for certain medications"],
        "economic_sensitivity": "Low sensitivity to economic cycles, high regulatory dependence"
    },
    "Electronics": {
        "description": "Egypt's electronics sector focuses on assembly operations, component manufacturing, and regional distribution.",
        "recommendation_focus": "Component suppliers, logistics companies, and tech retail businesses.",
        "key_regions": ["Greater Cairo", "Alexandria", "Suez Canal Zone"],
        "seasonal_factors": ["Holiday seasons affect consumer electronics demand"],
        "economic_sensitivity": "High sensitivity to global supply chain disruptions and import costs"
    },
    "Tourism-Related Products": {
        "description": "Businesses producing souvenirs, handicrafts, and tourism-related goods are significant in Egypt's economy.",
        "recommendation_focus": "Hotels, tour operators, and transportation companies in tourist-heavy regions.",
        "key_regions": ["Luxor", "Aswan", "Red Sea", "South Sinai", "Cairo"],
        "seasonal_factors": ["Winter tourism season (October-March) is peak demand", "European holiday seasons affect tourism patterns"],
        "economic_sensitivity": "Very high sensitivity to global tourism trends and security perceptions"
    }
}

# Regional Business Characteristics
REGIONAL_CHARACTERISTICS = {
    "Greater Cairo": {
        "business_density": "Very High",
        "infrastructure": "Excellent",
        "export_access": "Good access to airports and nearby ports",
        "typical_businesses": ["Manufacturing", "Services", "Technology", "Wholesale"],
        "recommendation_focus": "Urban market access, business services, and manufacturing partnerships"
    },
    "Mediterranean Coast": {
        "business_density": "High",
        "infrastructure": "Good",
        "export_access": "Excellent port access (Alexandria, Port Said)",
        "typical_businesses": ["Import/Export", "Shipping", "Seafood", "Tourism"],
        "recommendation_focus": "Export-oriented businesses, shipping partnerships, and port services"
    },
    "Suez Canal": {
        "business_density": "Medium",
        "infrastructure": "Good",
        "export_access": "Excellent access to shipping routes",
        "typical_businesses": ["Logistics", "Shipping Services", "Industrial", "Free Zone Operations"],
        "recommendation_focus": "International trade partners, logistics services, and industrial suppliers"
    },
    "Nile Delta": {
        "business_density": "High",
        "infrastructure": "Moderate",
        "export_access": "Good access to Alexandria and Damietta ports",
        "typical_businesses": ["Agriculture", "Textiles", "Food Processing"],
        "recommendation_focus": "Agricultural partnerships, food processing equipment, and textile manufacturing"
    },
    "Upper Egypt": {
        "business_density": "Low to Medium",
        "infrastructure": "Basic to Moderate",
        "export_access": "Limited direct access, relies on transportation to Cairo/Alexandria",
        "typical_businesses": ["Agriculture", "Tourism", "Handicrafts", "Mining"],
        "recommendation_focus": "Agricultural production, tourism services, and traditional crafts"
    },
    "Red Sea": {
        "business_density": "Medium",
        "infrastructure": "Good (in tourist areas)",
        "export_access": "Good access to ports and MENA markets",
        "typical_businesses": ["Tourism", "Hospitality", "Real Estate", "Diving & Recreation"],
        "recommendation_focus": "Tourism services, hospitality supplies, and recreational equipment"
    },
    "Sinai": {
        "business_density": "Low (except South Sinai tourist areas)",
        "infrastructure": "Variable (good in tourist areas, basic elsewhere)",
        "export_access": "Access to Red Sea and Mediterranean through Suez Canal",
        "typical_businesses": ["Tourism", "Agriculture", "Bedouin Crafts"],
        "recommendation_focus": "Tourism partnerships, craft marketing, and agricultural suppliers"
    }
}

# Egyptian Business Seasonality Factors
SEASONALITY_FACTORS = {
    "Tourism Seasons": {
        "winter_peak": "October to March - Higher demand for tourism-related goods and services",
        "summer_low": "June to August - Lower European tourism, higher Gulf tourism",
        "spring_moderate": "April to May - Moderate tourism levels",
        "fall_building": "September - Building toward winter peak season"
    },
    "Ramadan": {
        "description": "The Islamic holy month shifts each year (10-11 days earlier annually)",
        "business_impact": "Increased food sales before and during, reduced working hours, increased night-time consumer activity",
        "post_ramadan": "Eid al-Fitr brings increased clothing sales, gifts, and travel"
    },
    "Agricultural Cycles": {
        "cotton_harvest": "September to November - Peak cotton-related business activity",
        "citrus_export": "December to February - Peak citrus export season",
        "date_harvest": "September to October - Date processing and export activity",
        "wheat_harvest": "April to June - Grain storage and processing activity"
    },
    "Educational Calendar": {
        "back_to_school": "September to October - Increased demand for educational supplies",
        "exam_periods": "December, May - Specific educational materials and services",
        "summer_vacation": "June to August - Youth activities and family travel"
    }
}

# Recommendation System Technical Knowledge
RECOMMENDATION_TECHNICAL = {
    "hybrid_approach_explanation": """
        The Buy-From-Egypt recommendation system uses a hybrid approach that combines three methods:
        
        1. Collaborative Filtering: Analyzes past user behavior and finds patterns in purchase history
           to recommend products that similar users have bought.
           
        2. Content-Based Filtering: Recommends products and business partners based on specific 
           attributes of businesses, such as industry, region, and trade patterns.
           
        3. Contextual Awareness: Incorporates Egyptian economic indicators and seasonal factors
           to adjust recommendations based on current market conditions.
        
        This hybrid approach overcomes the limitations of any single method and provides more relevant
        recommendations for the unique Egyptian business environment.
    """,
    
    "api_integration": """
        To integrate with the recommendation API:
        
        1. Customer recommendations: Call /recommend/customer/{customer_id} to get personalized
           product recommendations for specific customers.
           
        2. Business recommendations: Call /recommend/business/{business_name} to get both product
           and potential business partnership recommendations.
           
        3. Economic context: Access current Egyptian economic context via /egyptian-economic-context
           to understand factors influencing recommendations.
           
        4. Data synchronization: Keep the recommendation system updated with your latest business data
           using the /sync endpoints (user, product, order).
           
        Detailed API documentation is available at the /api/docs endpoint.
    """,
    
    "recommendation_metrics": """
        The recommendation system's performance is measured using several metrics:
        
        - Precision@10: 0.398 (39.8% of recommended items are relevant)
        - Recall@10: 0.998 (99.8% of relevant items are recommended)
        - F1@10: 0.569 (Balanced score between precision and recall)
        - Business Recommendation Precision@5: 0.681 (68.1% of recommended partnerships are relevant)
        
        These metrics indicate the system provides relevant recommendations for Egyptian businesses,
        with particularly strong performance in business partnership recommendations.
    """,
    
    "data_requirements": """
        For optimal recommendations, the system needs:
        
        1. User profiles: Customer information including purchase history and preferences
        2. Business profiles: Company information including industry, location, and trade patterns
        3. Transaction data: Historical purchase data showing what products users have bought
        4. Product catalog: Detailed product information including categories and attributes
        
        The more data provided, the more accurate and personalized the recommendations will be.
        Data can be synchronized using the /sync API endpoints.
    """
}

# Common business questions and detailed answers
COMMON_BUSINESS_QUESTIONS = {
    "How can recommendations help my Egyptian business?": """
        The Buy-From-Egypt recommendation system can help your business in several ways:
        
        1. Discover new products to sell based on your business profile, industry trends, and 
           what similar Egyptian businesses are successfully offering
           
        2. Identify potential business partners in complementary industries, located in optimal
           regions for your supply chain or with access to markets you want to reach
           
        3. Understand how economic factors and seasonality in Egypt affect your business, with
           recommendations that adapt to these changing conditions
           
        4. Improve your product offerings based on data about what Egyptian customers are buying,
           especially in your specific industry and region
           
        5. Expand your market reach by connecting with businesses that have access to different
           regions or customer segments within Egypt
        
        The system is specifically designed for the Egyptian market, incorporating unique factors
        like regional business patterns, seasonal events like Ramadan, and traditional Egyptian
        industry strengths.
    """,
    
    "What makes these recommendations specific to Egypt?": """
        The recommendation system is tailored specifically for Egypt through several features:
        
        1. Regional understanding: Incorporates knowledge of Egypt's diverse regions (Greater Cairo,
           Mediterranean Coast, Upper Egypt, etc.) and their business characteristics
           
        2. Egyptian seasonality: Accounts for unique seasonal factors like winter tourism season,
           Ramadan, and Egyptian agricultural cycles
           
        3. Traditional strengths: Gives higher relevance to products where Egypt has historical
           advantages (textiles, agriculture, handicrafts, etc.)
           
        4. Economic context: Incorporates Egyptian economic indicators like GDP growth, inflation,
           and tourism sensitivity
           
        5. Market access: Considers proximity to target markets important for Egyptian businesses
           (Europe, MENA, Africa, Asia)
           
        6. Logistics awareness: Factors in access to ports, transportation hubs, and trade routes
           within Egypt
        
        This Egypt-specific approach ensures recommendations are practical and relevant for the
        local business environment rather than generic global suggestions.
    """,
    
    "How accurate is the recommendation system?": """
        The Buy-From-Egypt recommendation system achieves strong accuracy metrics:
        
        - 39.8% precision for top-10 recommendations, meaning about 4 out of 10 recommended
          products are highly relevant
          
        - 68.1% precision for business partnership recommendations, meaning more than 2 out of 3
          suggested business partners are genuinely good matches
          
        - 99.8% recall for relevant products, meaning the system successfully identifies nearly
          all products that would be relevant to a user
        
        These metrics have been validated through testing with actual Egyptian business data.
        The system performs particularly well for businesses with complete profile information
        and transaction history.
        
        The hybrid approach (combining collaborative filtering, content-based recommendations,
        and Egyptian context) significantly outperforms any single recommendation method.
    """,
    
    "How do I integrate these recommendations into my website or app?": """
        Integrating Buy-From-Egypt recommendations into your digital platforms is straightforward:
        
        1. API Integration:
           - Use our REST API endpoints to fetch recommendations
           - Key endpoints: /recommend/customer/{id} and /recommend/business/{name}
           - Full API documentation available at /api/docs
        
        2. Data Synchronization:
           - Keep your user, product, and order data synchronized with our system
           - Use the /sync endpoints to push new data
           - Regular synchronization improves recommendation quality
        
        3. Display Options:
           - Product carousels ("Recommended for you", "Customers also bought")
           - Business partner suggestions ("Potential business matches")
           - Export recommendations to CSV/JSON for your marketing campaigns
        
        4. Context Integration:
           - Use the /egyptian-economic-context endpoint to display relevant market insights
           - Adjust your UI based on seasonal factors (Ramadan, tourism seasons)
        
        Our team can provide implementation examples and technical support for integration
        with common e-commerce and business platforms.
    """,
    
    "What data do you need from my business?": """
        To provide optimal recommendations, we need the following data:
        
        1. Essential Business Information:
           - Business name and unique identifier
           - Industry category and subcategories
           - Location/region within Egypt
           - Business type (importer, exporter, or both)
           - Size of business (small, medium, large)
        
        2. Product Information:
           - Product catalog with unique identifiers
           - Product descriptions, categories, and attributes
           - Pricing information
           - Inventory availability
        
        3. Transaction Data:
           - Order history (which customers bought which products)
           - Order dates and values
           - Customer identifiers (anonymized if needed)
        
        4. Optional Enhancements:
           - Target markets (domestic regions or export countries)
           - Seasonal business patterns
           - Business partnerships and supply chain information
        
        You can provide this data through our secure API endpoints. All data is processed
        in compliance with privacy regulations and used solely to improve recommendation quality.
    """
}

# Export all knowledge for use in the chatbot
EGYPTIAN_BUSINESS_KNOWLEDGE = {
    "industry_sectors": INDUSTRY_SECTORS,
    "regional_characteristics": REGIONAL_CHARACTERISTICS,
    "seasonality_factors": SEASONALITY_FACTORS,
    "recommendation_technical": RECOMMENDATION_TECHNICAL,
    "common_business_questions": COMMON_BUSINESS_QUESTIONS
} 