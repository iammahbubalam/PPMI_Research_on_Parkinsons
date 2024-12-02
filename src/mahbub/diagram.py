import matplotlib.pyplot as plt

# Data
comparisons = ['V1 vs V2', 'V1 vs V3', 'V1 vs V4', 'V2 vs V3', 'V2 vs V4', 'V3 vs V4']
percentages = [63.1, 71.4, 77.4, 13.1, 63.1, 50.0]

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(comparisons, percentages, color='skyblue')

# Add title and labels
plt.title('Percentage of Features Above Significance Level (5%)')
plt.xlabel('Comparisons')
plt.ylabel('Percentage (%)')
# Show plot
plt.show()
# Data
comparisons = [
    'Minimal --> Mild', 'Minimal --> Moderate', 'Minimal --> Moderate to Severe',
    'Minimal --> Severe', 'Minimal --> Very Severe', 'Mild --> Moderate',
    'Mild --> Moderate to Severe', 'Mild --> Severe', 'Mild --> Very Severe',
    'Moderate --> Moderate to Severe', 'Moderate --> Severe',
    'Moderate --> Very Severe', 'Moderate to Severe --> Severe',
    'Moderate to Severe --> Very Severe', 'Severe --> Very Severe'
]
percentages = [
    12.50, 0.00, 29.17, 8.33, 50.00, 0.00, 33.33, 58.33, 62.50,
    33.33, 29.17, 58.33, 0.00, 45.83, 33.33
]

# Create bar chart
plt.figure(figsize=(12, 8))
plt.barh(comparisons, percentages, color='skyblue')

# Add title and labels
plt.title('Percentage of Image Features Above Significance Level (5%) in Each Disease Severity Class')
plt.xlabel('Percentage (%)')
plt.ylabel('Comparisons')

# Show plot
plt.tight_layout()
plt.show()

comparisons = ['V1 vs V2', 'V1 vs V3', 'V1 vs V4', 'V2 vs V3', 'V2 vs V4', 'V3 vs V4']
percentages = [33.3, 45.8, 45.8, 0.0, 12.5, 0.0]

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(comparisons, percentages, color='skyblue')

# Add title and labels
plt.title('Percentage of Image Features Above Significance Level (5%) in Each Visit')
plt.xlabel('Comparisons')
plt.ylabel('Percentage (%)')

# Show plot
plt.show()


# Data
comparisons = [
    'Mild --> Minimal', 'Mild --> Moderate', 'Mild --> Moderate to Severe', 'Mild --> Severe', 'Mild --> Very Severe',
    'Minimal --> Moderate', 'Minimal --> Moderate to Severe', 'Minimal --> Severe', 'Minimal --> Very Severe',
    'Moderate --> Moderate to Severe', 'Moderate --> Severe', 'Moderate --> Very Severe',
    'Moderate to Severe --> Severe', 'Moderate to Severe --> Very Severe', 'Severe --> Very Severe'
]
percentages = [
    73.33, 100.00, 100.00, 100.00, 93.33, 96.67, 100.00, 100.00, 96.67,
    93.33, 96.67, 93.33, 53.33, 93.33, 66.67
]

# Create bar chart
plt.figure(figsize=(12, 8))
plt.barh(comparisons, percentages, color='skyblue')

# Add title and labels
plt.title('Percentage of Clinical Features Above Significance Level (5%) in Each Disease Severity Class')
plt.xlabel('Percentage (%)')
plt.ylabel('Comparisons')

# Show plot
plt.tight_layout()
plt.show()

# Data
comparisons = ['V1 vs V2', 'V1 vs V3', 'V1 vs V4', 'V2 vs V3', 'V2 vs V4', 'V3 vs V4']
percentages = [33.3, 63.3, 83.3, 0.0, 56.7, 50.0]

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(comparisons, percentages, color='skyblue')

# Add title and labels
plt.title('Percentage of Clinical Features Above Significance Level (5%) in Each Visit')
plt.xlabel('Comparisons')
plt.ylabel('Percentage (%)')

# Show plot
plt.show()