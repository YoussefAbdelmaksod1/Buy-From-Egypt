#!/usr/bin/env python3
"""
Buy From Egypt - Business Recommendation System
Professional interface for Egyptian business recommendations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

# Add src to path for importing models
sys.path.append(str(Path(__file__).parent / "src"))

# Page configuration
st.set_page_config(
    page_title="Buy From Egypt",
    page_icon="🇪🇬",
    layout="wide"
)

@st.cache_resource
def load_recommendation_engine():
    """Load the recommendation engine"""
    try:
        from models.inference import load_recommendation_engine
        engine = load_recommendation_engine()
        return engine
    except Exception as e:
        st.error(f"Error loading recommendation models: {e}")
        return None

def main():
    # Header
    st.title("Buy From Egypt")
    st.markdown("**Business Recommendation System for Egyptian Commerce**")
    
    # Load recommendation engine
    with st.spinner("Loading recommendation system..."):
        engine = load_recommendation_engine()
    
    if engine is None:
        st.error("Failed to load recommendation system. Please check configuration.")
        return
    
    # Check model status
    model_status = engine.get_model_status()
    
    if model_status['content_model_loaded']:
        st.success("Recommendation system is active")
    else:
        st.warning("Using basic recommendation system")
    
    # Main workflow tabs
    tab1, tab2 = st.tabs(["Register Business", "Get Recommendations"])
    
    with tab1:
        st.header("Register Your Egyptian Business")
        st.markdown("Register your business to receive personalized product recommendations")
        
        with st.form("business_registration"):
            col1, col2 = st.columns(2)
            
            with col1:
                business_name = st.text_input("Business Name *", placeholder="e.g., Cairo Premium Textiles")
                business_id = st.text_input("Business ID *", placeholder="e.g., cairo_textiles_2025")
                email = st.text_input("Email *", placeholder="e.g., info@business.eg")
                country = st.text_input("Country", value="Egypt")
            
            with col2:
                business_type = st.selectbox("Business Type *", ["EXPORTER", "IMPORTER", "MANUFACTURER"])
                industry = st.selectbox("Industry Sector *", [
                    "Textiles", "Agriculture", "Tourism", "Electronics", 
                    "Handicrafts", "Food Processing", "Chemicals", "Construction",
                    "Pharmaceuticals", "Metals", "Furniture"
                ])
                region = st.selectbox("Region", [
                    "Greater Cairo", "Nile Delta", "Upper Egypt", "Mediterranean Coast",
                    "Red Sea Coast", "Suez Canal", "Western Desert"
                ])
                active = st.checkbox("Active Business", value=True)
            
            submitted = st.form_submit_button("Register Business", type="primary")
            
            if submitted:
                if business_name and business_id and email and industry:
                    business_data = {
                        "userId": business_id,
                        "name": business_name,
                        "email": email,
                        "type": business_type,
                        "industrySector": industry,
                        "country": country,
                        "region": region,
                        "active": active
                    }
                    
                    # Store in session state
                    st.session_state['registered_business'] = business_data
                    st.session_state['business_name'] = business_name
                    st.session_state['industry'] = industry
                    
                    st.success(f"{business_name} registered successfully!")
                    st.json(business_data)
                else:
                    st.error("Please fill in all required fields marked with *")
    
    with tab2:
        st.header("Get Product Recommendations")
        st.markdown("Receive personalized product recommendations based on your business profile")
        
        if 'registered_business' not in st.session_state:
            st.warning("Please register your business first in the 'Register Business' tab")
            return
        
        business_data = st.session_state['registered_business']
        st.info(f"Getting recommendations for: **{business_data['name']}** ({business_data['industrySector']})")
        
        # Find similar business for content-based recommendations
        similar_businesses = {
            "Textiles": "El Fabrics Textiles",
            "Agriculture": "International Fruits Agriculture Egypt", 
            "Food Processing": "Grains Agriculture Group",
            "Tourism": "International Souvenirs Tourism Egypt",
            "Handicrafts": "International Souvenirs Tourism Egypt",
            "Electronics": "Premium Electronics Egypt",
            "Construction": "Cairo Construction Materials",
            "Pharmaceuticals": "Egyptian Pharmaceuticals Co",
            "Metals": "Silver Metals Trading Egypt"
        }
        
        similar_business = similar_businesses.get(business_data['industrySector'], "El Fabrics Textiles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Content-Based Recommendations")
            st.write("Based on businesses similar to yours in the same industry")
            st.write(f"Analyzing businesses similar to: **{similar_business}**")
            
            num_products = st.slider("Number of Product Recommendations:", 5, 20, 10)
            
            if st.button("Generate Product Recommendations", type="primary"):
                with st.spinner("Generating recommendations..."):
                    recommendations = engine.recommend_products_for_business(
                        similar_business, num_products
                    )
                    
                    if recommendations:
                        st.session_state['content_recommendations'] = recommendations
                        
                        st.success(f"Generated {len(recommendations)} product recommendations!")
                        
                        # Show top 5 recommendations
                        st.write("**Top 5 Product Recommendations:**")
                        for i, product in enumerate(recommendations[:5], 1):
                            st.write(f"{i}. **{product['StockCode']}** - {product['Description']}")
                        
                        # Category breakdown
                        products_df = pd.DataFrame(recommendations)
                        categories = [desc.split()[0] for desc in products_df['Description']]
                        category_counts = pd.Series(categories).value_counts()
                        
                        fig = px.pie(values=category_counts.values, names=category_counts.index,
                                   title=f"Recommended Categories for {business_data['industrySector']}")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("No recommendations generated")
        
        with col2:
            st.subheader("Collaborative Filtering")
            st.write("Based on customer behavior patterns and preferences")
            
            customer_id = st.selectbox("Analyze Customer Pattern:", [
                "10001 - Electronics & Food Processing",
                "10002 - Mixed Preferences",
                "10003 - Construction Focus", 
                "10004 - Agriculture Interest",
                "10005 - Tourism Products"
            ])
            
            customer_id = customer_id.split(" - ")[0]
            
            if st.button("Generate Customer-Based Recommendations", type="primary"):
                with st.spinner("Analyzing customer behavior..."):
                    customer_recs = engine.recommend_products_for_customer(customer_id, 10)
                    
                    if customer_recs:
                        st.session_state['collaborative_recommendations'] = customer_recs
                        
                        st.success(f"Generated {len(customer_recs)} customer-based recommendations!")
                        
                        # Show top 5 recommendations
                        st.write("**Top 5 Customer-Based Recommendations:**")
                        for i, product in enumerate(customer_recs[:5], 1):
                            st.write(f"{i}. **{product['StockCode']}** - {product['Description']}")
                        
                        # Score distribution
                        products_df = pd.DataFrame(customer_recs)
                        fig = px.bar(products_df.head(8), x='StockCode', y='Score',
                                   title="Customer Preference Scores")
                        fig.update_xaxes(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("No customer recommendations generated")
        
        # Show combined results if both are available
        if 'content_recommendations' in st.session_state and 'collaborative_recommendations' in st.session_state:
            st.subheader("Combined Recommendations Analysis")
            
            content_recs = st.session_state['content_recommendations']
            collab_recs = st.session_state['collaborative_recommendations']
            
            # Find common products
            content_products = set(p['StockCode'] for p in content_recs)
            collab_products = set(p['StockCode'] for p in collab_recs)
            common_products = content_products.intersection(collab_products)
            
            if common_products:
                st.success(f"**{len(common_products)} products** recommended by both approaches!")
                st.write("**High-Confidence Recommendations:**")
                for product_code in list(common_products)[:3]:
                    content_product = next(p for p in content_recs if p['StockCode'] == product_code)
                    st.write(f"• **{product_code}** - {content_product['Description']}")
            else:
                st.info("Different approaches suggest different opportunities - consider diversification")
            
            # Business recommendations
            st.write("**Recommended Next Steps:**")
            st.write("1. Focus on high-confidence products from both recommendation types")
            st.write("2. Research market demand for suggested product categories")
            st.write("3. Analyze competitors in recommended product areas")
            st.write("4. Consider seasonal factors for Egyptian market timing")
            st.write("5. Use export data to plan international expansion")

if __name__ == "__main__":
    main()
