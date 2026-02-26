import csv
import statistics
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import random

# Set global font to Arial
mpl.rcParams['font.family'] = 'Arial'

# Read CSV file and analyze data by category
def analyze_results(csv_file):
    # Store performance improvement percentages by category and algorithm
    cost_improvements = {
        'large_dense': {'greedy_001': [], 'greedy_002': [], 'greedy_003': []},
        'large_sparse': {'greedy_001': [], 'greedy_002': [], 'greedy_003': []},
        'small_dense': {'greedy_001': [], 'greedy_002': [], 'greedy_003': []},
        'small_sparse': {'greedy_001': [], 'greedy_002': [], 'greedy_003': []},
        'fifth_category': {'greedy_001': [], 'greedy_002': [], 'greedy_003': []}
    }
    
    time_improvements = {
        'large_dense': {'greedy_001': [], 'greedy_002': [], 'greedy_003': []},
        'large_sparse': {'greedy_001': [], 'greedy_002': [], 'greedy_003': []},
        'small_dense': {'greedy_001': [], 'greedy_002': [], 'greedy_003': []},
        'small_sparse': {'greedy_001': [], 'greedy_002': [], 'greedy_003': []},
        'fifth_category': {'greedy_001': [], 'greedy_002': [], 'greedy_003': []}
    }
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Get dataset category
                category = row.get('category', 'unknown')
                if category not in cost_improvements:
                    category = 'unknown'
                
                # Get random algorithm time and cost
                random_time = float(row['random_algorithm_time'])
                random_cost = float(row['random_algorithm_cost'])
                
                # Calculate performance improvement for each greedy algorithm
                for algo in ['greedy_001', 'greedy_002', 'greedy_003']:
                    algo_time = float(row[f'{algo}_time'])
                    algo_cost = float(row[f'{algo}_cost'])
                    
                    # Calculate cost improvement percentage (how much it decreased)
                    if random_cost > 0:
                        cost_improvement = ((random_cost - algo_cost) / random_cost) * 100
                        cost_improvements[category][algo].append(cost_improvement)
                    
                    # Calculate time improvement percentage (how much it shortened)
                    if random_time > 0:
                        time_improvement = ((random_time - algo_time) / random_time) * 100
                        time_improvements[category][algo].append(time_improvement)
        
        # Calculate averages by category
        print("\n======================================")
        print("Analysis Results by Dataset Category")
        print("======================================")
        
        categories = ['large_dense', 'large_sparse', 'small_dense', 'small_sparse', 'fifth_category']
        
        for category in categories:
            print(f"\nCategory: {category}")
            print("-" * 50)
            
            for algo in ['greedy_001', 'greedy_002', 'greedy_003']:
                print(f"  {algo}:")
                
                # Average cost improvement
                if cost_improvements[category][algo]:
                    avg_cost_improvement = statistics.mean(cost_improvements[category][algo])
                    print(f"    Average cost reduction: {avg_cost_improvement:.2f}%")
                else:
                    print("    Average cost reduction: No data")
                
                # Average time improvement
                if time_improvements[category][algo]:
                    avg_time_improvement = statistics.mean(time_improvements[category][algo])
                    print(f"    Average time reduction: {avg_time_improvement:.2f}%")
                else:
                    print("    Average time reduction: No data")
        
        print("\n======================================")
        print("Analysis completed")
        print("======================================")
        
    except FileNotFoundError:
        print(f"Error: File not found {csv_file}")
    except Exception as e:
        print(f"Analysis error: {e}")

# Visualize algorithm cost reduction performance
def visualize_cost_performance(csv_file):
    # Store data points
    data = {
        'elements_count': [],
        'greedy_001': [],
        'greedy_002': [],
        'greedy_003': []
    }
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Get element count
                elements_count = int(row['elements_count'])
                data['elements_count'].append(elements_count)
                
                # Get random algorithm cost
                random_cost = float(row['random_algorithm_cost'])
                
                # Calculate cost improvement percentage for each greedy algorithm
                for algo in ['greedy_001', 'greedy_002', 'greedy_003']:
                    algo_cost = float(row[f'{algo}_cost'])
                    if random_cost > 0:
                        cost_improvement = ((random_cost - algo_cost) / random_cost) * 100
                        data[algo].append(cost_improvement)
                    else:
                        data[algo].append(0)  # Avoid division by zero
        
        # Create scatter plot
        plt.figure(figsize=(12, 6))
        
        # Plot data points for each algorithm
        plt.scatter(data['elements_count'], data['greedy_001'], color='blue', label='greedy_001', alpha=0.6)
        plt.scatter(data['elements_count'], data['greedy_002'], color='green', label='greedy_002', alpha=0.6)
        plt.scatter(data['elements_count'], data['greedy_003'], color='red', label='greedy_003', alpha=0.6)
        
        # Add title and labels
        plt.title('Algorithm Cost Reduction vs Element Count', fontname='Arial')
        plt.xlabel('Element Count', fontname='Arial')
        plt.ylabel('Cost Reduction Percentage vs Random Algorithm (%)', fontname='Arial')
        
        # Set x-axis to log scale for better visualization of large ranges
        plt.xscale('log')
        
        # Add legend
        plt.legend(prop={'family': 'Arial'})
        
        # Add grid lines
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        output_file = 'cost_performance_visualization.png'
        plt.savefig(output_file, dpi=150)
        print(f"\nCost plot saved to: {output_file}")
        
        # Show plot
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: File not found {csv_file}")
    except Exception as e:
        print(f"Cost visualization error: {e}")

# Visualize algorithm time reduction performance
def visualize_time_performance(csv_file):
    # Store data points
    data = {
        'elements_count': [],
        'greedy_001': [],
        'greedy_002': [],
        'greedy_003': []
    }
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Get element count
                elements_count = int(row['elements_count'])
                data['elements_count'].append(elements_count)
                
                # Get random algorithm time
                random_time = float(row['random_algorithm_time'])
                
                # Calculate time improvement percentage for each greedy algorithm
                for algo in ['greedy_001', 'greedy_002', 'greedy_003']:
                    algo_time = float(row[f'{algo}_time'])
                    if random_time > 0:
                        time_improvement = ((random_time - algo_time) / random_time) * 100
                        data[algo].append(time_improvement)
                    else:
                        data[algo].append(0)  # Avoid division by zero
        
        # Create scatter plot
        plt.figure(figsize=(12, 6))
        
        # Plot data points for each algorithm
        plt.scatter(data['elements_count'], data['greedy_001'], color='blue', label='greedy_001', alpha=0.6)
        plt.scatter(data['elements_count'], data['greedy_002'], color='green', label='greedy_002', alpha=0.6)
        plt.scatter(data['elements_count'], data['greedy_003'], color='red', label='greedy_003', alpha=0.6)
        
        # Add title and labels
        plt.title('Algorithm Time Reduction vs Element Count', fontname='Arial')
        plt.xlabel('Element Count', fontname='Arial')
        plt.ylabel('Time Reduction Percentage vs Random Algorithm (%)', fontname='Arial')
        
        # Set x-axis to log scale for better visualization of large ranges
        plt.xscale('log')
        
        # Add legend
        plt.legend(prop={'family': 'Arial'})
        
        # Add grid lines
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        output_file = 'time_performance_visualization.png'
        plt.savefig(output_file, dpi=150)
        print(f"\nTime plot saved to: {output_file}")
        
        # Show plot
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: File not found {csv_file}")
    except Exception as e:
        print(f"Time visualization error: {e}")

# Classify datasets based on size and density
def classify_datasets(csv_file, data_dir):
    """
    Classify datasets into categories based on:
    1. Element count (large vs small)
    2. Subset density (dense vs sparse)
    3. Randomly select 20% as fifth category
    
    Update CSV with classification results and return category statistics
    """
    # Read existing data
    rows = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)
        
        if not rows:
            print("Error: No data found in CSV file")
            return
        
        # Calculate element count threshold (median)
        element_counts = [int(row['elements_count']) for row in rows]
        size_threshold = statistics.median(element_counts)
        
        # Calculate subset size for each dataset and determine density threshold
        dataset_info = []
        for row in rows:
            file_name = row['file_name']
            file_path = os.path.join(data_dir, file_name)
            
            # Read data file to calculate subset sizes
            try:
                with open(file_path, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                
                if not lines:
                    print(f"Warning: Empty data file {file_name}")
                    avg_subset_size = 0
                else:
                    # First line: n_elements, n_sets
                    first_line = lines[0].split()
                    n_sets = int(first_line[1]) if len(first_line) > 1 else 0
                    
                    # Calculate average subset size
                    total_elements = 0
                    valid_sets = 0
                    
                    for line_idx in range(1, min(len(lines), n_sets + 1)):
                        parts = lines[line_idx].split()
                        if len(parts) > 1:  # At least cost + one element
                            total_elements += len(parts) - 1
                            valid_sets += 1
                    
                    avg_subset_size = total_elements / valid_sets if valid_sets > 0 else 0
            except Exception as e:
                print(f"Error reading data file {file_name}: {e}")
                avg_subset_size = 0
            
            dataset_info.append((row, avg_subset_size))
        
        # Calculate density threshold (median of average subset sizes)
        subset_sizes = [info[1] for info in dataset_info if info[1] > 0]
        density_threshold = statistics.median(subset_sizes) if subset_sizes else 0
        
        # Randomly select 20% of datasets as fifth category
        random.seed(42)  # For reproducibility
        fifth_category_count = max(1, int(len(rows) * 0.2))
        fifth_category_indices = random.sample(range(len(rows)), fifth_category_count)
        
        # Classify datasets and update CSV
        category_stats = {
            'large_dense': {'element_counts': [], 'subset_sizes': [], 'count': 0},
            'large_sparse': {'element_counts': [], 'subset_sizes': [], 'count': 0},
            'small_dense': {'element_counts': [], 'subset_sizes': [], 'count': 0},
            'small_sparse': {'element_counts': [], 'subset_sizes': [], 'count': 0},
            'fifth_category': {'element_counts': [], 'subset_sizes': [], 'count': 0}
        }
        
        # Add classification field to fieldnames if not exists
        if 'category' not in fieldnames:
            fieldnames.append('category')
        
        # Classify each dataset
        for idx, (row, avg_subset_size) in enumerate(dataset_info):
            element_count = int(row['elements_count'])
            
            # Check if this dataset is in fifth category
            if idx in fifth_category_indices:
                category = 'fifth_category'
            else:
                # Determine size category
                size_cat = 'large' if element_count >= size_threshold else 'small'
                # Determine density category
                density_cat = 'dense' if avg_subset_size >= density_threshold else 'sparse'
                category = f'{size_cat}_{density_cat}'
            
            # Update row with category
            row['category'] = category
            
            # Update category stats
            category_stats[category]['element_counts'].append(element_count)
            category_stats[category]['subset_sizes'].append(avg_subset_size)
            category_stats[category]['count'] += 1
        
        # Write updated data back to CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row, _ in dataset_info:
                writer.writerow(row)
        
        # Print category statistics
        print("\n======================================")
        print("Dataset Classification Statistics")
        print("======================================")
        print(f"Size threshold (elements): {size_threshold:.2f}")
        print(f"Density threshold (avg subset size): {density_threshold:.2f}")
        print()
        
        for category, stats in category_stats.items():
            if stats['count'] > 0:
                avg_elements = statistics.mean(stats['element_counts'])
                avg_subset = statistics.mean(stats['subset_sizes'])
            else:
                avg_elements = 0
                avg_subset = 0
            
            print(f"{category}:")
            print(f"  Sample count: {stats['count']}")
            print(f"  Average elements: {avg_elements:.2f}")
            print(f"  Average subset size: {avg_subset:.2f}")
            print()
        
        print("======================================")
        print("Classification completed and CSV updated")
        print("======================================")
        
    except FileNotFoundError:
        print(f"Error: File not found {csv_file}")
    except Exception as e:
        print(f"Classification error: {e}")

if __name__ == "__main__":
    # CSV file path
    csv_file = "setcover_results.csv"
    data_dir = "./data"
    
    # First classify datasets
    # classify_datasets(csv_file, data_dir)
    
    # Then analyze results and visualize
    analyze_results(csv_file)
    # visualize_cost_performance(csv_file)
    # visualize_time_performance(csv_file)