import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
RETAIL_DATA = DATA_DIR / "data.csv"
BUSINESS_DATA = DATA_DIR / "enhanced_egypt_import_export_v2.csv"
ECONOMIC_DATA = DATA_DIR / "12906e42-6096-445d-a44d-052bca7256eb_Data.csv"

# Create processed directory if it doesn't exist
PROCESSED_DIR.mkdir(exist_ok=True)

def clean_retail_data():
    """
    Clean and preprocess the retail transaction data.
    Focus on Egyptian context by:
    1. Prioritizing transactions in Egypt if available
    2. Adding Egyptian seasonality factors
    """
    logger.info("Loading retail transaction data...")
    try:
        df = pd.read_csv(RETAIL_DATA, encoding='ISO-8859-1')
        
        # Data cleaning steps
        logger.info("Cleaning retail data and applying Egyptian context...")
        
        # Remove rows with missing CustomerID
        df = df.dropna(subset=['CustomerID'])
        
        # Convert CustomerID to integer and then to string
        df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
        
        # Convert InvoiceDate to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Add month and year columns
        df['Month'] = df['InvoiceDate'].dt.month
        df['Year'] = df['InvoiceDate'].dt.year
        
        # Add transaction value
        df['TransactionValue'] = df['Quantity'] * df['UnitPrice']
        
        # Filter out entries with negative quantity (returns)
        df = df[df['Quantity'] > 0]
        
        # Filter out entries with zero unit price
        df = df[df['UnitPrice'] > 0]
        
        # EGYPT-SPECIFIC MODIFICATIONS:
        
        # Filter for Egypt transactions if available
        egypt_transactions = df[df['Country'] == 'Egypt']
        if len(egypt_transactions) > 100:  # If sufficient Egypt transactions exist
            logger.info(f"Found {len(egypt_transactions)} transactions from Egypt. Using Egypt-only data.")
            df = egypt_transactions
        else:
            logger.info("Insufficient Egypt-specific transactions. Adding Egyptian context to global data.")
            
            # Add Egyptian seasonality factor (tourism seasons, Ramadan, etc.)
            # Higher demand during winter tourism season (Oct-Mar)
            winter_months = [10, 11, 12, 1, 2, 3]
            df['IsWinterTourism'] = df['Month'].isin(winter_months).astype(int)
            
            # Ramadan (shifting dates - using approximation)
            # In real implementation, would use actual Ramadan dates per year
            ramadan_months_by_year = {
                2010: [8, 9],  # August-September 2010
                2011: [8],     # August 2011
                2012: [7, 8],  # July-August 2012
                2013: [7],     # July 2013
                2014: [6, 7],  # June-July 2014
                2015: [6],     # June 2015
                2016: [6],     # June 2016
                2017: [5, 6],  # May-June 2017
                2018: [5],     # May 2018
                2019: [5],     # May 2019
                2020: [4, 5],  # April-May 2020
                2021: [4],     # April 2021
                2022: [4],     # April 2022
                2023: [3, 4],  # March-April 2023
            }
            
            # Apply Ramadan context
            df['IsRamadan'] = 0
            for year, months in ramadan_months_by_year.items():
                mask = (df['Year'] == year) & (df['Month'].isin(months))
                df.loc[mask, 'IsRamadan'] = 1
                
            # Add priority for Egyptian-relevant products
            # These are keywords relevant to Egyptian markets
            egyptian_keywords = [
                'cotton', 'textile', 'spice', 'craft', 'ceramic', 'papyrus',
                'leather', 'copper', 'silver', 'gold', 'carpet', 'rug',
                'dates', 'olive', 'tea', 'coffee', 'lamp', 'glass', 'metal',
                'furniture', 'decoration', 'ornament', 'jewelry', 'herb'
            ]
            
            # Create a column indicating Egyptian relevance
            df['EgyptRelevance'] = df['Description'].str.lower().apply(
                lambda desc: 1 if any(keyword in str(desc).lower() for keyword in egyptian_keywords) else 0
            )
            
            # Add slight boost to items that may be relevant to Egyptian market
            df['AdjustedValue'] = df['TransactionValue'] * (1 + df['EgyptRelevance'] * 0.2)
        
        # Save cleaned data
        df.to_csv(PROCESSED_DIR / "retail_cleaned.csv", index=False)
        logger.info(f"Cleaned retail data with Egyptian context saved to {PROCESSED_DIR}/retail_cleaned.csv")
        
        return df
    
    except Exception as e:
        logger.error(f"Error cleaning retail data: {e}")
        raise

def clean_business_data():
    """
    Clean and preprocess the business profile data.
    Enhance with Egyptian business context.
    """
    logger.info("Loading business profile data...")
    try:
        df = pd.read_csv(BUSINESS_DATA)
        
        # Data cleaning steps
        logger.info("Cleaning business data and adding Egyptian business context...")
        
        # Remove any duplicate business entries
        df = df.drop_duplicates(subset=['Business Name'])
        
        # Fill missing values or handle them as appropriate
        if 'Trade Growth Rate (%)' in df.columns:
            df['Trade Growth Rate (%)'].fillna(df['Trade Growth Rate (%)'].median(), inplace=True)
        else:
            df['Trade Growth Rate (%)'] = 5.0  # Default value if column doesn't exist
        
        # Check if Price Fluctuation column exists
        if 'Price Fluctuation (%)' in df.columns:
            df['Price Fluctuation (%)'].fillna(0, inplace=True)
        else:
            df['Price Fluctuation (%)'] = 0.0  # Default value if column doesn't exist
        
        # Create a unique business ID
        df['BusinessID'] = range(1, len(df) + 1)
        
        # Convert categorical variables
        df['Trade Type'] = df['Trade Type'].astype('category')
        df['Business Size'] = df['Business Size'].astype('category')
        
        # EGYPT-SPECIFIC MODIFICATIONS:
        
        # 1. Add Egyptian governorate regions from location data
        # Extract governorate from location if possible
        def extract_governorate(location):
            if pd.isna(location):
                return "Unknown"
            
            location = location.lower()
            governorates = {
                'cairo': 'Greater Cairo',
                'giza': 'Greater Cairo',
                'alexandria': 'Mediterranean Coast',
                'luxor': 'Upper Egypt',
                'aswan': 'Upper Egypt',
                'hurghada': 'Red Sea',
                'sharm': 'Sinai',
                'sinai': 'Sinai',
                'suez': 'Suez Canal',
                'ismailia': 'Suez Canal',
                'port said': 'Mediterranean Coast',
                'mansoura': 'Nile Delta',
                'tanta': 'Nile Delta',
                'damietta': 'Mediterranean Coast',
                'fayoum': 'Upper Egypt',
                'minya': 'Upper Egypt',
                'qena': 'Upper Egypt',
                'sohag': 'Upper Egypt',
                'beni suef': 'Upper Egypt',
                'assiut': 'Upper Egypt'
            }
            
            for key, region in governorates.items():
                if key in location:
                    return region
            
            return "Other Egypt"
        
        df['Region'] = df['Location'].apply(extract_governorate)
        
        # 2. Add proximity to major ports/logistics hubs
        logistics_hubs = {
            'Greater Cairo': 4,  # Good access to major logistics
            'Mediterranean Coast': 5,  # Excellent port access
            'Suez Canal': 5,  # Excellent port access
            'Red Sea': 4,  # Good port access
            'Upper Egypt': 2,  # Limited access to ports
            'Nile Delta': 3,  # Moderate logistics access
            'Sinai': 3,  # Moderate access to ports
            'Other Egypt': 2   # Limited logistics
        }
        
        df['LogisticsAccess'] = df['Region'].map(logistics_hubs)
        
        # 3. Add proximity to target markets
        market_access = {
            'Mediterranean Coast': {
                'Europe': 5,
                'MENA': 4,
                'Africa': 3,
                'Asia': 3
            },
            'Suez Canal': {
                'Europe': 4,
                'MENA': 5, 
                'Africa': 3,
                'Asia': 5
            },
            'Greater Cairo': {
                'Europe': 3,
                'MENA': 4,
                'Africa': 4,
                'Asia': 3
            },
            'Red Sea': {
                'Europe': 3,
                'MENA': 5,
                'Africa': 4, 
                'Asia': 4
            },
            'Upper Egypt': {
                'Europe': 2,
                'MENA': 3,
                'Africa': 5,
                'Asia': 2
            },
            'Nile Delta': {
                'Europe': 4,
                'MENA': 3, 
                'Africa': 3,
                'Asia': 3
            },
            'Sinai': {
                'Europe': 3,
                'MENA': 5,
                'Africa': 3,
                'Asia': 4
            },
            'Other Egypt': {
                'Europe': 3,
                'MENA': 3,
                'Africa': 3,
                'Asia': 3
            }
        }
        
        # Based on main trading partner, determine target market
        def get_target_market(partner):
            if pd.isna(partner):
                return "MENA"  # Default
                
            partner = str(partner).lower()
            
            europe = ['germany', 'france', 'italy', 'spain', 'uk', 'greece', 'netherlands', 'belgium', 'sweden', 'poland']
            mena = ['saudi', 'uae', 'qatar', 'kuwait', 'oman', 'bahrain', 'jordan', 'lebanon', 'iraq', 'turkey']
            africa = ['nigeria', 'kenya', 'ethiopia', 'south africa', 'ghana', 'algeria', 'morocco', 'tunisia', 'libya']
            asia = ['china', 'india', 'japan', 'korea', 'malaysia', 'indonesia', 'singapore', 'vietnam', 'thailand']
            
            for country in europe:
                if country in partner:
                    return "Europe"
            for country in mena:
                if country in partner:
                    return "MENA"
            for country in africa:
                if country in partner:
                    return "Africa"
            for country in asia:
                if country in partner:
                    return "Asia"
                    
            return "MENA"  # Default if no match
            
        df['TargetMarket'] = df['Main Trading Partner'].apply(get_target_market)
        
        # Calculate market access score
        df['MarketAccessScore'] = df.apply(
            lambda row: market_access[row['Region']][row['TargetMarket']], 
            axis=1
        )
        
        # 4. Calculate an "Egyptian Advantage Score" combining:
        # - Traditional Egyptian industry (higher score for traditional Egyptian exports)
        # - Logistics access
        # - Target market access
        
        traditional_industry_scores = {
            'Textiles': 5,     # Strong Egyptian cotton industry
            'Agriculture': 5,  # Key Egyptian export
            'Spices': 5,       # Traditional strong sector
            'Fruits & Vegetables': 5,  # Strong Egyptian export
            'Chemicals': 4,    # Growing Egyptian sector
            'Pharmaceuticals': 4,  # Growing Egyptian sector
            'Electronics': 3,  # Emerging sector
            'Machinery': 3,    # Moderate Egyptian presence
            'Metals': 4,       # Good Egyptian industry
            'Automobiles': 2,  # Limited Egyptian advantage
            'Seafood': 4,      # Good sector with access to Mediterranean/Red Sea
            'Manufacturing': 3 # Moderate Egyptian advantage
        }
        
        df['TraditionalIndustryScore'] = df['Category'].map(traditional_industry_scores).fillna(3)
        
        # Calculate overall Egyptian advantage score
        df['EgyptianAdvantageScore'] = (
            df['TraditionalIndustryScore'] * 0.4 +  # Industry advantage
            df['LogisticsAccess'] * 0.3 +           # Logistics advantage
            df['MarketAccessScore'] * 0.3           # Market access advantage
        )
        
        # Create feature vectors for each business
        df['Business Features'] = df.apply(
            lambda row: {
                'trade_volume': row['Annual Trade Volume (M USD)'],
                'growth_rate': row['Trade Growth Rate (%)'],
                'success_rate': row['Trade Success Rate (%)'],
                'trade_frequency': row['Trade Frequency (per year)'],
                'business_size': row['Business Size'],
                'trade_type': row['Trade Type'],
                'category': row['Category'],
                'egyptian_advantage': row['EgyptianAdvantageScore'],
                'location_region': row['Region']
            }, axis=1
        )
        
        # Save cleaned data
        df.to_csv(PROCESSED_DIR / "business_cleaned.csv", index=False)
        logger.info(f"Cleaned business data with Egyptian context saved to {PROCESSED_DIR}/business_cleaned.csv")
        
        return df
    
    except Exception as e:
        logger.error(f"Error cleaning business data: {e}")
        raise

def clean_economic_data():
    """
    Clean and preprocess the economic indicators data.
    Focus on Egyptian indicators.
    """
    logger.info("Loading economic indicators data...")
    try:
        df = pd.read_csv(ECONOMIC_DATA)
        
        # Data cleaning steps
        logger.info("Cleaning economic data with focus on Egypt-relevant indicators...")
        
        # Filter for Egypt
        df = df[df['Country Name'] == 'Egypt, Arab Rep.']
        
        # Get the year columns (they start with a digit)
        year_columns = [col for col in df.columns if col[0].isdigit() or 'YR' in col]
        
        logger.info(f"Found {len(df)} economic indicators for Egypt.")
        
        # Reshape data from wide to long format (unpivot)
        df_long = pd.melt(
            df,
            id_vars=['Country Name', 'Series Name'],
            value_vars=year_columns,
            var_name='Year',
            value_name='Value'
        )
        
        # Extract the year from the column name (format: "2000 [YR2000]")
        df_long['Year'] = df_long['Year'].str.extract(r'(\d{4})').astype(int)
        
        # Replace '..' with NaN
        df_long['Value'] = df_long['Value'].replace('..', np.nan)
        
        # Convert to numeric, errors='coerce' will convert non-numeric values to NaN
        df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')
        
        # Only keep the most recent years with data (last 10 years)
        recent_years = sorted(df_long['Year'].unique())[-10:]
        df_long = df_long[df_long['Year'].isin(recent_years)]
        
        # Pivot to get indicators as columns
        df_pivoted = df_long.pivot_table(
            index='Year',
            columns='Series Name',
            values='Value',
            aggfunc='mean'  # Will work now that we have numeric values
        )
        
        # Key Egyptian economic indicators
        key_indicators = [
            'GDP growth (annual %)',
            'Inflation, consumer prices (annual %)',
            'Population growth (annual %)',
            'Unemployment, total (% of total labor force) (national estimate)'
        ]
        
        # Get the latest economic indicators
        latest_year = max(df_pivoted.index)
        latest_economic = df_pivoted.loc[latest_year].to_dict()
        
        # Create a more accessible subset of key indicators
        key_economic = {k: latest_economic.get(k) for k in key_indicators if k in latest_economic}
        
        logger.info(f"Found {len(key_economic)} key Egyptian economic indicators.")
        
        # Save processed data
        df_pivoted.to_csv(PROCESSED_DIR / "economic_indicators.csv")
        
        with open(PROCESSED_DIR / "latest_economic.json", 'w') as f:
            import json
            json.dump(latest_economic, f, indent=2)
        
        logger.info(f"Cleaned economic data saved to {PROCESSED_DIR}/economic_indicators.csv")
        
        # Add Egyptian seasonality and context
        # Create a dictionary with Egyptian-specific economic context
        egyptian_context = {
            'gdp_growth': key_economic.get('GDP growth (annual %)', 4.35),
            'inflation': key_economic.get('Inflation, consumer prices (annual %)', 5.04),
            'population_growth': key_economic.get('Population growth (annual %)', 1.73),
            'tourism_sensitivity': 0.85,  # High sensitivity to tourism (scale 0-1)
            'economic_stability_index': 0.65,  # Medium-high stability (scale 0-1)
            'trade_balance': -0.12,  # Trade deficit as proportion of GDP
            'is_winter_tourism_season': 1 if datetime.now().month in [10, 11, 12, 1, 2, 3] else 0,
            'is_ramadan_season': 0  # Would be calculated based on Islamic calendar
        }
        
        # Save Egyptian context
        with open(PROCESSED_DIR / "egyptian_context.json", 'w') as f:
            json.dump(egyptian_context, f, indent=2)
        
        return df_pivoted, latest_economic
    
    except Exception as e:
        logger.error(f"Error cleaning economic data: {e}")
        raise

def prepare_interaction_matrix():
    """
    Create a user-item interaction matrix from the retail data.
    """
    logger.info("Creating user-item interaction matrix...")
    try:
        # Load cleaned retail data
        retail_df = pd.read_csv(PROCESSED_DIR / "retail_cleaned.csv")
        
        # Create user-item interaction matrix
        # We'll use CustomerID as user and StockCode as item
        # The value will be the sum of quantities purchased
        
        user_item_matrix = retail_df.groupby(['CustomerID', 'StockCode'])['Quantity'].sum().unstack().fillna(0)
        
        # Create a binary interaction matrix (1 if user purchased item, 0 otherwise)
        binary_matrix = (user_item_matrix > 0).astype(int)
        
        # Save matrices
        user_item_matrix.to_csv(PROCESSED_DIR / "user_item_matrix.csv")
        binary_matrix.to_csv(PROCESSED_DIR / "binary_interaction_matrix.csv")
        
        logger.info(f"User-item matrices saved to {PROCESSED_DIR}")
        
        return user_item_matrix, binary_matrix
    
    except Exception as e:
        logger.error(f"Error creating interaction matrix: {e}")
        raise

def prepare_business_features():
    """
    Create feature vectors for businesses.
    """
    logger.info("Preparing business feature vectors...")
    try:
        # Load cleaned business data
        business_df = pd.read_csv(PROCESSED_DIR / "business_cleaned.csv")
        
        # Create a mapping of categorical variables
        trade_type_map = {'Importer': 0, 'Exporter': 1, 'Both': 2}
        business_size_map = {'Small': 0, 'Medium': 1, 'Large': 2}
        
        # Get unique categories and create mapping
        categories = business_df['Category'].unique()
        category_map = {cat: i for i, cat in enumerate(categories)}
        
        # Create feature vectors
        features = business_df[['BusinessID', 'Business Name', 'Annual Trade Volume (M USD)', 
                              'Trade Growth Rate (%)', 'Trade Success Rate (%)', 
                              'Trade Frequency (per year)', 'Business Size', 
                              'Trade Type', 'Category']].copy()
        
        # Apply mappings
        features['Trade Type Encoded'] = features['Trade Type'].map(trade_type_map)
        features['Business Size Encoded'] = features['Business Size'].map(business_size_map)
        features['Category Encoded'] = features['Category'].map(category_map)
        
        # Save feature vectors
        features.to_csv(PROCESSED_DIR / "business_features.csv", index=False)
        
        # Save category mapping for reference
        pd.DataFrame(list(category_map.items()), columns=['Category', 'Encoded']).to_csv(
            PROCESSED_DIR / "category_mapping.csv", index=False
        )
        
        logger.info(f"Business feature vectors saved to {PROCESSED_DIR}/business_features.csv")
        
        return features
    
    except Exception as e:
        logger.error(f"Error preparing business features: {e}")
        raise

def main():
    """
    Main function to execute all data processing steps.
    """
    logger.info("Starting data processing...")
    
    try:
        # Clean and preprocess datasets
        retail_df = clean_retail_data()
        business_df = clean_business_data()
        economic_df, latest_economic = clean_economic_data()
        
        # Create derived datasets
        user_item_matrix, binary_matrix = prepare_interaction_matrix()
        business_features = prepare_business_features()
        
        logger.info("Data processing completed successfully.")
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise

if __name__ == "__main__":
    main() 
# Skip function tests - directly run the API
if __name__ == '__main__': exit(0)
