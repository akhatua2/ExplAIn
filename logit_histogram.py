import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(data, num_bins, save_path, x_label, y_label, title, exclude_last_bin=False):
    """
    Plot and save a histogram of the given numpy array
    
    Args:
        data: numpy array of values to plot
        num_bins: number of bins for the histogram
        save_path: path to save the output PNG file
        x_label: label for x-axis
        y_label: label for y-axis
        title: title of the plot
        exclude_last_bin: if True, excludes the last bin from plotting
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate the range of the data
    min_val, max_val = np.min(data), np.max(data)
    
    if exclude_last_bin:
        # Calculate bin edges
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        # Find the threshold for the second-to-last bin
        threshold = bin_edges[-2]
        # Filter out data that would go into the last bin
        filtered_data = data[data < threshold]
        # Plot histogram with filtered data
        plt.hist(filtered_data, bins=num_bins-1, edgecolor='black', range=(min_val, threshold))
    else:
        # Plot normal histogram with all data
        plt.hist(data, bins=num_bins, edgecolor='black', range=(min_val, max_val))
    
    # Customize the plot
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    # Format x-axis to show actual values
    plt.xticks(rotation=45)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_histogram_focus(data, num_bins, save_path, x_label, y_label, title, focus_range_percentile=90):
    """
    Plot and save a histogram of the given numpy array, focusing on the main distribution
    by removing outliers based on percentile range.
    
    Args:
        data: numpy array of values to plot
        num_bins: number of bins for the histogram
        save_path: path to save the output PNG file
        x_label: label for x-axis
        y_label: label for y-axis
        title: title of the plot
        focus_range_percentile: percentile range to focus on (e.g., 90 means look at 5th to 95th percentile)
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate the percentile range to focus on
    lower_percentile = (100 - focus_range_percentile) / 2
    upper_percentile = 100 - lower_percentile
    
    # Get the value range for the focused region
    min_val = np.percentile(data, lower_percentile)
    max_val = np.percentile(data, upper_percentile)
    
    # Filter data to only include values within the focus range
    filtered_data = data[(data >= min_val) & (data <= max_val)]
    
    # Create histogram with filtered data
    n, bins, patches = plt.hist(filtered_data, bins=num_bins, edgecolor='black', range=(min_val, max_val))
    
    # Add information about the focus range to the title
    full_title = f"{title}\n(Showing {focus_range_percentile}% of data: {lower_percentile:.1f}th to {upper_percentile:.1f}th percentile)"
    
    # Customize the plot
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(full_title)
    
    # Format x-axis to show actual values
    plt.xticks(rotation=45)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_histogram_top(data, num_bins, save_path, x_label, y_label, title, start_percentile=10):
    """
    Plot and save a histogram of the given numpy array, showing data from start_percentile to max
    
    Args:
        data: numpy array of values to plot
        num_bins: number of bins for the histogram
        save_path: path to save the output PNG file
        x_label: label for x-axis
        y_label: label for y-axis
        title: title of the plot
        start_percentile: percentile to start from (e.g., 10 means show top 90% of data)
    """
    plt.figure(figsize=(12, 6))
    
    # Get the value at the start percentile
    min_val = np.percentile(data, start_percentile)
    max_val = np.max(data)
    
    # Filter data to only include values above the start percentile
    filtered_data = data[data >= min_val]
    
    # Create histogram with filtered data
    n, bins, patches = plt.hist(filtered_data, bins=num_bins, edgecolor='black', range=(min_val, max_val))
    
    # Add information about the range to the title
    full_title = f"{title}\n(Showing top {100-start_percentile}% of data: >{start_percentile}th percentile)"
    
    # Customize the plot
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(full_title)
    
    # Format x-axis to show actual values
    plt.xticks(rotation=45)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# save_name = "grpo_logits.npz"
save_name = "pretrained_logits.npz"
arr = []
dic = np.load(save_name)
for key in dic.keys():
    arr.append(dic[key])

arr = np.concatenate(arr, axis=-1)

# print(arr[0])

# assert False
plot_histogram(arr, num_bins=100, save_path="pre_logits_histogram_exclude.png", x_label="Value", y_label="Frequency", title=  "Logits Histogram", exclude_last_bin=False)

# dic = np.load(save_name)

# print(dic['1'].shape)

# Example usage with focus on main distribution
# plot_histogram_focus(arr, 
#                     num_bins=100, 
#                     save_path="grpo_logits_histogram_focused.png", 
#                     x_label="Value", 
#                     y_label="Frequency", 
#                     title="Logits Histogram",
#                     focus_range_percentile=90)

# Example usage showing top 90% of data
# plot_histogram_top(arr, 
#                   num_bins=100, 
#                   save_path="grpo_logits_histogram_top80.png", 
#                   x_label="Value", 
#                   y_label="Frequency", 
#                   title="Logits Histogram",
#                   start_percentile=20)



