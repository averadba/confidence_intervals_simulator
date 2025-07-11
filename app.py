import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to calculate confidence intervals
def calculate_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    margin_of_error = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, mean - margin_of_error, mean + margin_of_error

# Function to plot the confidence intervals
def plot_confidence_intervals(sample_means, confidence_intervals, population_mean):
    plt.figure(figsize=(10,6))
    
    # Convert confidence intervals to error bars format
    error_bars = [(mean - ci[1], ci[2] - mean) for mean, ci in zip(sample_means, confidence_intervals)]
    
    # Unpack the lower and upper bounds
    lower_bounds, upper_bounds = zip(*error_bars)
    
    # Plot the sample means and confidence intervals
    for i, mean in enumerate(sample_means):
        plt.errorbar(i, mean, yerr=[[lower_bounds[i]], [upper_bounds[i]]], fmt='o')
    
    # Plot the population mean
    plt.hlines(population_mean, xmin=-1, xmax=len(sample_means), colors='yellow', label='Population Mean')
    
    # Formatting
    plt.title('Sampling Distribution and Confidence Intervals')
    plt.xlabel('Sample Number')
    plt.ylabel('Sample Mean')
    plt.legend()
    
    plt.tight_layout()
    
    # Tell Streamlit to display the Matplotlib plot.
    st.pyplot(plt)

# Streamlit app
st.title('Confidence Interval Simulator')
st.markdown("By: [Alexis Vera](mailto:alexisvera@gmail.com)")


# Display instructions and explanations
st.write("""
         This simulator generates a number of samples from a population with a specified mean and standard deviation.
         It calculates the confidence intervals for the mean of each sample and plots them.
         The table below the plot shows the total number of intervals generated and how many of those do not contain the population mean.
         You can adjust the population parameters, sample size, number of samples, and confidence level using the sidebar controls.
         Press the 'Generate Confidence Intervals' button to update the plot and the table.
         """)


# Sidebar controls
confidence_level = st.sidebar.slider('Confidence Level', 90, 99, 95)
population_mean = st.sidebar.number_input('Population Mean', value=0.0)
population_std_dev = st.sidebar.number_input('Population Std Dev', value=1.0)
sample_size = st.sidebar.number_input('Sample Size', value=30)
number_of_samples = st.sidebar.number_input('Number of Samples', value=100)

# Generate population data
population_data = np.random.normal(population_mean, population_std_dev, 10000)

# Generate samples and calculate sample means and confidence intervals
sample_means = []
confidence_intervals = []
intervals_not_containing_mean = 0

for _ in range(number_of_samples):
    sample = np.random.choice(population_data, size=sample_size, replace=False)
    sample_mean = np.mean(sample)
    sample_means.append(sample_mean)
    ci = calculate_confidence_interval(sample, confidence=confidence_level/100)
    confidence_intervals.append(ci)
    
    # Check if the population mean is not within the confidence interval
    if ci[1] > population_mean or ci[2] < population_mean:
        intervals_not_containing_mean += 1

# Plotting
if st.button('Generate Confidence Intervals'):
    plot_confidence_intervals(sample_means, confidence_intervals, population_mean)
    
    # Create a DataFrame to display the table
    data = {
        "Total Intervals Generated": [number_of_samples],
        "Intervals Not Containing Mean": [intervals_not_containing_mean]
    }
    results_df = pd.DataFrame(data)
    st.write(results_df)
