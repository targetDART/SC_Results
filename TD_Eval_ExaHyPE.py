import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys

dirs = [ 'sched', '1980'] #'sched'
measurements = ['plain', 'TD_GPU', 'TD_ANY', 'TD_GPU_multi', 'TD_ANY_multi']
subfolder = '8'
filename = '0.txt'

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname + '/ExaHyPE_output')

values = dict()

def plot_values_bar_chart_with_text(values_dict, 
                                     dir_rename_map=None, 
                                     measurement_rename_map=None,
                                     x_axis_title="Configuration", 
                                     y_axis_title="Finite Volume Updates / Second", 
                                     legend_title="Measurement Method",
                                     plot_title="Performance Comparison",
                                     text_format='{:,.0f}'): 
    if not values_dict:
        print("Warning: Input dictionary is empty. Cannot generate plot.")
        return None

    # --- 1. Convert dictionary to DataFrame ---
    try:
        df = pd.DataFrame.from_dict(values_dict, orient='index')
        df = df.fillna(value=np.nan) 
        
        if df.empty:
             print("Warning: DataFrame created from dictionary is empty. Cannot generate plot.")
             return None

        df = df.reset_index() 
        df = df.rename(columns={'index': '_original_dir'}) 
        
        # --- 2. Melt DataFrame to long format ---
        measurement_cols = [col for col in df.columns if col != '_original_dir']
        if not measurement_cols:
            print("Warning: No measurement columns found in DataFrame. Cannot generate plot.")
            return None

        df_long = pd.melt(df, 
                          id_vars=['_original_dir'], 
                          value_vars=measurement_cols, 
                          var_name='_original_measurement', 
                          value_name='_value')
                          
    except Exception as e:
        print(f"Error converting data to DataFrame or melting: {e}")
        return None

    # --- 3. Apply Renaming ---
    df_long['display_dir'] = df_long['_original_dir'].map(dir_rename_map or {}).fillna(df_long['_original_dir'])
    df_long['display_measurement'] = df_long['_original_measurement'].map(measurement_rename_map or {}).fillna(df_long['_original_measurement'])

    # --- 4. Create Formatted Text for Labels ---
    if text_format:
        try:
            df_long['_text_value'] = df_long['_value'].apply(
            lambda x: f"{x / 1_000_000:.2f}M" if pd.notna(x) else None
            )
            text_column_name = '_text_value' 
        except Exception as e:
            print(f"Warning: Could not apply text formatting to millions: {e}. Disabling text labels.", file=sys.stderr)
            text_column_name = None 
    else:
        text_column_name = None 

    # --- 5. Create the Plot ---
    try:
        fig = px.bar(df_long, 
                     x='display_dir', 
                     y='_value', 
                     color='display_measurement',
                     barmode='group', 
                     title=plot_title,
                     labels={ 
                         'display_dir': x_axis_title,
                         '_value': y_axis_title,
                         'display_measurement': legend_title 
                     },
                     text=text_column_name, 
                     template='plotly_white' 
                    )
        
        # --- 6. Update Trace Layout (Text Position and Color) --- # MODIFIED HERE
        if text_column_name:
            fig.update_traces(
                textposition='outside',   # MODIFIED: Always place text above the bar
                textfont_color='black',   # ADDED: Set text color to black
                textfont_size=10          # Optional: Adjust size
                # cliponaxis=False        # Optional: Prevents text from being clipped by plot edges
            ) 

        # --- 7. Update Overall Layout (Legend Position, etc.) --- 
        fig.update_layout(
            yaxis_title=y_axis_title,
            xaxis_title=x_axis_title,
            legend_title_text=legend_title, 
            legend=dict(             
                x=1.1,                 
                y=1,                 
                xanchor='right',     
                yanchor='top',       
            )
        )

        # Add margin at the top to prevent text cutoff (adjust value as needed)
        # This gives extra space above the highest bar/text for 'outside' text position
        fig.update_layout(yaxis_range=[0, df_long['_value'].max() * 1.15]) # Increase top by 15%

        return fig
        
    except Exception as e:
        print(f"Error creating Plotly bar chart: {e}")
        return None
   
def parse_cell_updates(filename):
    target_string = "Average cell updates per second:"
    last_value = None

    try:
        with open(filename, 'r') as f:
            for line in f:
                # Remove leading/trailing whitespace
                stripped_line = line.strip() 
                
                if target_string in stripped_line:
                    try:
                        # Extract the part after the target string
                        value_str = stripped_line.split(target_string)[-1].strip()
                        # Attempt to convert to float and store it
                        last_value = float(value_str) * 1024
                    except (ValueError, IndexError):
                        # Handle cases where splitting fails or conversion to float fails
                        # Continue searching for a later valid line
                        print(f"Warning: Found target string but could not parse float value in line: {line.strip()}", file=sys.stderr)
                        pass 
                        
    except FileNotFoundError:
        print(f"Error: File not found at '{filename}'", file=sys.stderr)
        raise # Re-raise the exception
    except IOError as e:
        print(f"Error reading file '{filename}': {e}", file=sys.stderr)
        raise # Re-raise the exception
        
    if last_value is None:
        print(f"Warning: Target string '{target_string}' not found or no valid float value followed it in file '{filename}'", file=sys.stderr)

    return last_value

for dir in dirs:
    values[dir] = dict()
    for measurement in measurements:
        filepath = os.path.join(dir, measurement, subfolder, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                output = file.read()
                print(f"Reading file: {filepath}")
                # Extract the value using the parsing function
                value = parse_cell_updates(filepath)
                values[dir][measurement] = value
        else:
            raise FileNotFoundError(f"File not found: {filepath}")

dir_display_names = {
        'sched': 'Ill-Balanced',
        '1980': 'Well-Balanced' 
    }
    
measurement_display_names = {
    'plain': 'Baseline OpenMP target',
    'TD_GPU': 'TARGETDART_OFFLOAD',
    'TD_ANY': 'TARGETDART_ANY',
    'TD_GPU_multi': 'TARGETDART_OFFLOAD multi',
    'TD_ANY_multi': 'TARGETDART_ANY multi'
}

figure = plot_values_bar_chart_with_text(
        values_dict=values,
        dir_rename_map=dir_display_names,
        measurement_rename_map=measurement_display_names,
        x_axis_title="Load Distribution", # Custom X axis title
        y_axis_title="Cell Updates / Second", # Custom Y axis title
        legend_title="", #"Offloading Method", # Custom legend title
        plot_title="", #"ExaHyPE Cell Update Performance" # Custom plot title
    )

name = "ExaHyPE"
if figure:
    try:
        # Make sure you have kaleido installed: pip install -U kaleido
        figure.write_image(f'{name}.svg') 
        print(f"Figure saved as {name}.svg")
        os.replace(f'{name}.svg', f'../{name}.svg')
    except Exception as e:
        print(f"Error saving figure as SVG: {e}")
        print("Please ensure 'kaleido' is installed ('pip install -U kaleido')")