import pandas as pd
import numpy as np
import os
from pathlib import Path
import random
from datetime import datetime, timedelta

# Define paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

# Define Egyptian cities and regions
egyptian_cities = {
    'Cairo': 'Greater Cairo',
    'Giza': 'Greater Cairo',
    '6th of October': 'Greater Cairo',
    'New Cairo': 'Greater Cairo',
    'Alexandria': 'Mediterranean Coast',
    'Port Said': 'Mediterranean Coast',
    'Damietta': 'Mediterranean Coast',
    'Marsa Matruh': 'Mediterranean Coast',
    'Suez': 'Suez Canal',
    'Ismailia': 'Suez Canal',
    'Port Tawfik': 'Suez Canal',
    'Sharm El Sheikh': 'Sinai',
    'Dahab': 'Sinai',
    'El Arish': 'Sinai',
    'Hurghada': 'Red Sea',
    'Safaga': 'Red Sea',
    'Luxor': 'Upper Egypt',
    'Aswan': 'Upper Egypt',
    'Qena': 'Upper Egypt',
    'Sohag': 'Upper Egypt',
    'Assiut': 'Upper Egypt',
    'Minya': 'Upper Egypt',
    'Fayoum': 'Upper Egypt',
    'Beni Suef': 'Upper Egypt',
    'Tanta': 'Nile Delta',
    'Mansoura': 'Nile Delta',
    'Zagazig': 'Nile Delta',
    'Damanhur': 'Nile Delta',
    'Kafr El Sheikh': 'Nile Delta'
}

# Define Egyptian industry sectors
industry_sectors = {
    'Textiles': ['Cotton', 'Garments', 'Fabrics', 'Carpets', 'Rugs'],
    'Agriculture': ['Fruits', 'Vegetables', 'Grains', 'Flowers', 'Dates'],
    'Pharmaceuticals': ['Medicine', 'Medical Supplies', 'Herbal', 'Vitamins'],
    'Metals': ['Copper', 'Aluminum', 'Steel', 'Gold', 'Silver'],
    'Food Processing': ['Dairy', 'Beverages', 'Confectionery', 'Canned Foods', 'Spices'],
    'Chemicals': ['Fertilizers', 'Plastics', 'Detergents', 'Paints', 'Cosmetics'],
    'Electronics': ['Components', 'Appliances', 'Communications', 'Computers'],
    'Construction': ['Cement', 'Bricks', 'Marble', 'Glass', 'Ceramics'],
    'Tourism': ['Hotels', 'Resorts', 'Transportation', 'Souvenirs', 'Craft'],
    'Furniture': ['Wooden', 'Metal', 'Office', 'Home', 'Outdoor'],
    'Handicrafts': ['Papyrus', 'Pottery', 'Jewelry', 'Leather', 'Brassware']
}

# Trading partners by region
trading_partners = {
    'Europe': ['Germany', 'Italy', 'France', 'Spain', 'United Kingdom', 'Netherlands', 'Belgium', 'Greece'],
    'Middle East': ['Saudi Arabia', 'UAE', 'Kuwait', 'Qatar', 'Oman', 'Bahrain', 'Jordan', 'Lebanon'],
    'Africa': ['Nigeria', 'Kenya', 'South Africa', 'Morocco', 'Tunisia', 'Ethiopia', 'Ghana', 'Tanzania'],
    'Asia': ['China', 'India', 'Japan', 'South Korea', 'Malaysia', 'Indonesia', 'Singapore', 'Vietnam'],
    'Americas': ['United States', 'Canada', 'Brazil', 'Mexico', 'Argentina', 'Chile']
}

# Egyptian business name patterns
business_name_patterns = [
    "Egyptian {sector}",
    "Pharaoh {sector}",
    "Nile {sector}",
    "Cairo {sector}",
    "Alexandria {sector}",
    "Delta {sector}",
    "{city} {sector}",
    "El {sector}",
    "{sector} of Egypt",
    "Royal {sector}",
    "Ancient {sector}",
    "Modern {sector}",
    "Golden {sector}",
    "Premium {sector}",
    "{region} {sector}",
    "United {sector}",
    "International {sector} Egypt",
    "{sector} Trading Egypt",
    "{sector} Group",
    "{sector} Industries",
    "{sector} Exports",
    "Egypt {sector} Company",
    "{sector} House",
    "{sector} Egypt"
]

# Egyptian customer names (fictional)
egyptian_first_names = [
    "Ahmed", "Mohamed", "Mahmoud", "Ali", "Hassan", "Hussein", "Omar", "Khaled", "Mostafa", "Ibrahim",
    "Amr", "Youssef", "Tarek", "Sherif", "Karim", "Hossam", "Ayman", "Walid", "Sameh", "Ramy",
    "Fatma", "Aisha", "Nour", "Mariam", "Mona", "Hoda", "Amira", "Dina", "Rania", "Sarah",
    "Yasmin", "Eman", "Laila", "Heba", "Dalia", "Noha", "Salma", "Soha", "Rasha", "Ghada"
]

egyptian_last_names = [
    "Ibrahim", "Mohamed", "Ahmed", "Mahmoud", "Ali", "Hassan", "Hussein", "Saad", "Mostafa", "Elsayed",
    "Abdelrahman", "Elsherbiny", "Mansour", "Fouad", "Abdallah", "Salah", "Youssef", "Abouzeid", "Zaki", "Nasser",
    "Farouk", "Adel", "Soliman", "Fathy", "Abdelaziz", "Samy", "Magdy", "Samir", "Abdelhamid", "Gamal"
]

def generate_business_name(sector, subsector=None, city=None, region=None):
    """Generate a realistic Egyptian business name."""
    pattern = random.choice(business_name_patterns)
    if subsector:
        sector_text = f"{subsector} {sector}"
    else:
        sector_text = sector
        
    if "{city}" in pattern and city:
        return pattern.format(sector=sector_text, city=city, region=region)
    elif "{region}" in pattern and region:
        return pattern.format(sector=sector_text, city=city, region=region)
    else:
        return pattern.format(sector=sector_text)

def generate_enhanced_business_data(n_businesses=200):
    """Generate enhanced business data for Egypt."""
    businesses = []
    business_ids = set()
    
    for _ in range(n_businesses):
        # Select industry sector and subsector
        sector = random.choice(list(industry_sectors.keys()))
        subsector = random.choice(industry_sectors[sector])
        
        # Select location
        city = random.choice(list(egyptian_cities.keys()))
        region = egyptian_cities[city]
        
        # Generate business details
        business_name = generate_business_name(sector, subsector, city, region)
        
        # Ensure name uniqueness
        while business_name in business_ids:
            business_name = generate_business_name(sector, subsector, city, region)
        business_ids.add(business_name)
        
        # Select trade type
        trade_type = random.choices(
            ['Importer', 'Exporter', 'Both'], 
            weights=[0.3, 0.4, 0.3]
        )[0]
        
        # Determine likely trading region based on region and trade type
        if trade_type == 'Importer' or trade_type == 'Both':
            if region in ['Mediterranean Coast', 'Suez Canal']:
                likely_regions = ['Europe', 'Middle East', 'Asia']
            elif region in ['Upper Egypt', 'Nile Delta']:
                likely_regions = ['Africa', 'Middle East']
            else:
                likely_regions = list(trading_partners.keys())
        else:
            if sector in ['Textiles', 'Handicrafts', 'Furniture']:
                likely_regions = ['Europe', 'Americas', 'Middle East']
            elif sector in ['Agriculture', 'Food Processing']:
                likely_regions = ['Middle East', 'Europe', 'Africa']
            else:
                likely_regions = list(trading_partners.keys())
                
        trading_region = random.choices(
            likely_regions,
            weights=[1/len(likely_regions)] * len(likely_regions)
        )[0]
        
        # Select specific trading partner
        main_trading_partner = random.choice(trading_partners[trading_region])
        
        # Business size based on region and sector
        if region in ['Greater Cairo', 'Mediterranean Coast', 'Suez Canal']:
            size_weights = [0.3, 0.4, 0.3]  # Small, Medium, Large
        else:
            size_weights = [0.5, 0.3, 0.2]  # More small businesses outside major regions
            
        business_size = random.choices(
            ['Small', 'Medium', 'Large'],
            weights=size_weights
        )[0]
        
        # Annual trade volume based on size
        if business_size == 'Small':
            volume_base = random.uniform(0.5, 15)
        elif business_size == 'Medium':
            volume_base = random.uniform(10, 80)
        else:
            volume_base = random.uniform(50, 500)
            
        # Adjust volume based on sector and region
        if sector in ['Textiles', 'Agriculture', 'Tourism']:
            volume_multiplier = 1.2  # Traditional strong sectors
        else:
            volume_multiplier = 1.0
            
        if region in ['Greater Cairo', 'Mediterranean Coast', 'Suez Canal']:
            volume_multiplier *= 1.3  # Better infrastructure regions
            
        annual_trade_volume = round(volume_base * volume_multiplier, 2)
        
        # Generate other business metrics
        established_year = random.randint(1980, 2022)
        years_operating = 2025 - established_year
        
        growth_rate = min(35, max(-10, random.normalvariate(5, 8) + (10 - years_operating) * 0.5))
        trade_frequency = int(max(3, min(365, annual_trade_volume * 3 + random.normalvariate(12, 6))))
        success_rate = min(98, max(50, random.normalvariate(78, 10) + (business_size == 'Large') * 10))
        
        # Generate contact info
        email_domain = business_name.lower().replace(' ', '').replace('&', 'and')[:15] + '.eg'
        email = f"info@{email_domain}"
        
        phone = f"+20{random.randint(1, 3)}{random.randint(1000, 9999)}{random.randint(1000, 9999)}"
        
        # Add Egyptian specific details
        if region == 'Mediterranean Coast' or region == 'Suez Canal':
            preferred_port = city if random.random() < 0.7 else random.choice(['Alexandria', 'Port Said', 'Damietta', 'Suez'])
        else:
            preferred_port = random.choice(['Alexandria', 'Port Said', 'Damietta', 'Suez', 'Safaga'])
            
        # Quality rating by international partners
        if success_rate > 90:
            rating = random.uniform(4.5, 5.0)
        elif success_rate > 80:
            rating = random.uniform(4.0, 4.7)
        elif success_rate > 70:
            rating = random.uniform(3.5, 4.2)
        else:
            rating = random.uniform(2.5, 3.8)
            
        rating = round(rating, 1)
        
        # Create business entry
        business = {
            'Business Name': business_name,
            'Location': f"{city}, Egypt",
            'Category': sector,
            'Annual Trade Volume (M USD)': annual_trade_volume,
            'Main Trading Partner': main_trading_partner,
            'Trade Type': trade_type,
            'Subcategory': subsector,
            'Business Size': business_size,
            'Established Year': established_year,
            'Contact Info': email,
            'Phone': phone,
            'Preferred Trade Route': f"Via {preferred_port} Port",
            'Trade Growth Rate (%)': round(growth_rate, 2),
            'Currency': 'USD',
            'Trade Frequency (per year)': trade_frequency,
            'Recency of Last Trade (months)': random.randint(0, 6),
            'Trade Success Rate (%)': round(success_rate, 2),
            'Partner Rating': rating,
            'Price Fluctuation (%)': round(random.uniform(1, 15), 2),
            'Region': region,
            'IndustrySector': f"{sector} - {subsector}"
        }
        
        businesses.append(business)
    
    return pd.DataFrame(businesses)

def generate_customer_data(n_customers=500):
    """Generate realistic Egyptian customer data."""
    customers = []
    
    for i in range(1, n_customers + 1):
        first_name = random.choice(egyptian_first_names)
        last_name = random.choice(egyptian_last_names)
        
        # Generate purchase patterns
        purchases_per_month = random.randint(1, 20)
        avg_purchase_value = random.uniform(10, 500)
        
        preference = random.choice(['Local', 'Imported', 'Mixed'])
        if preference == 'Local':
            local_preference = random.uniform(0.7, 0.9)
        elif preference == 'Imported':
            local_preference = random.uniform(0.1, 0.4)
        else:
            local_preference = random.uniform(0.4, 0.7)
        
        # Location info
        city = random.choice(list(egyptian_cities.keys()))
        region = egyptian_cities[city]
        
        # Customer type and segment
        customer_type = random.choice(['Individual', 'Business', 'Business'])  # More businesses
        if customer_type == 'Business':
            segment = random.choice(['Retailer', 'Wholesaler', 'Distributor', 'Manufacturer'])
            company_name = f"{last_name} {random.choice(['Trading', 'Enterprises', 'Group', 'Company', 'Industries'])}"
        else:
            segment = random.choice(['Regular', 'Premium', 'Economy'])
            company_name = None
        
        # Create customer record
        customer = {
            'CustomerID': 10000 + i,
            'FirstName': first_name,
            'LastName': last_name,
            'CompanyName': company_name,
            'CustomerType': customer_type,
            'Segment': segment,
            'City': city,
            'Region': region,
            'Country': 'Egypt',
            'PurchasesPerMonth': purchases_per_month,
            'AvgPurchaseValue': round(avg_purchase_value, 2),
            'LocalPreference': round(local_preference, 2),
            'JoinDate': (datetime.now() - timedelta(days=random.randint(30, 3650))).strftime('%Y-%m-%d')
        }
        
        customers.append(customer)
    
    return pd.DataFrame(customers)

def generate_transaction_data(customers_df, products_df, n_transactions=5000):
    """Generate realistic transaction data between customers and products."""
    transactions = []
    
    customer_ids = customers_df['CustomerID'].tolist()
    
    # If products_df doesn't have all the fields we need, we'll create a subset of necessary fields
    if 'StockCode' not in products_df.columns:
        # Create simplified product data
        product_data = []
        for i in range(1000):
            category = random.choice(list(industry_sectors.keys()))
            subcategory = random.choice(industry_sectors[category])
            
            product = {
                'StockCode': f"EGY{i:05d}",
                'Description': f"{subcategory} {random.choice(['Product', 'Item', 'Good', 'Merchandise'])}",
                'Category': category,
                'Subcategory': subcategory,
                'UnitPrice': round(random.uniform(5, 500), 2),
                'Origin': random.choice(['Egypt', 'Imported'])
            }
            product_data.append(product)
        
        products_df = pd.DataFrame(product_data)
    
    # Generate transactions
    invoice_id = 10001
    for _ in range(n_transactions):
        # Select a customer
        customer_id = random.choice(customer_ids)
        customer = customers_df[customers_df['CustomerID'] == customer_id].iloc[0]
        
        # Transaction date
        days_ago = random.randint(1, 365 * 2)  # Up to 2 years of history
        transaction_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d %H:%M:%S')
        
        # Number of items in this transaction
        n_items = random.randint(1, 10)
        
        # Customer preferences affect product selection
        if 'LocalPreference' in customer:
            local_preference = customer['LocalPreference']
        else:
            local_preference = 0.5  # Default
        
        # Select products for this transaction
        for _ in range(n_items):
            # Decide if product should be local or imported based on customer preference
            if random.random() < local_preference and 'Origin' in products_df.columns:
                product_subset = products_df[products_df['Origin'] == 'Egypt']
                if len(product_subset) == 0:  # Fallback if no matching products
                    product_subset = products_df
            else:
                product_subset = products_df
            
            # Randomly select a product
            product = product_subset.iloc[random.randint(0, len(product_subset) - 1)]
            
            # Determine quantity and price
            quantity = random.randint(1, 20)
            unit_price = product['UnitPrice'] if 'UnitPrice' in product else random.uniform(5, 500)
            
            # Create transaction record
            transaction = {
                'InvoiceNo': f"INV{invoice_id}",
                'StockCode': product['StockCode'],
                'Description': product['Description'],
                'Quantity': quantity,
                'InvoiceDate': transaction_date,
                'UnitPrice': round(unit_price, 2),
                'CustomerID': customer_id,
                'Country': 'Egypt'
            }
            
            if 'Category' in product:
                transaction['Category'] = product['Category']
                
            if 'Subcategory' in product:
                transaction['Subcategory'] = product['Subcategory']
                
            transactions.append(transaction)
        
        invoice_id += 1
    
    return pd.DataFrame(transactions)

def main():
    """Generate enhanced Egyptian-specific data."""
    print("Generating enhanced Egyptian business data...")
    business_df = generate_enhanced_business_data(n_businesses=200)
    business_df.to_csv(DATA_DIR / "enhanced_egypt_import_export_v2.csv", index=False)
    print(f"Generated {len(business_df)} enhanced Egyptian business records")
    
    print("Generating Egyptian customer data...")
    customer_df = generate_customer_data(n_customers=500)
    customer_df.to_csv(DATA_DIR / "egyptian_customers.csv", index=False)
    print(f"Generated {len(customer_df)} Egyptian customer records")
    
    print("Generating transaction data...")
    # Create a simple product dataset if needed
    product_df = pd.DataFrame([{
        'StockCode': f"EGY{i:05d}",
        'Description': f"{random.choice(list(industry_sectors.keys()))} Product {i}",
        'UnitPrice': round(random.uniform(5, 500), 2),
        'Origin': random.choice(['Egypt', 'Imported'])
    } for i in range(1000)])
    
    # Generate transaction data
    transaction_df = generate_transaction_data(customer_df, product_df, n_transactions=10000)
    transaction_df.to_csv(DATA_DIR / "data.csv", index=False)
    print(f"Generated {len(transaction_df)} transaction records")
    
    print("All enhanced Egyptian data generation complete!")

if __name__ == "__main__":
    main() 