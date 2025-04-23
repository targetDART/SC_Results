import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

num_values = 20
dirs = ['Eval', 'Eval_scal', 'Eval_over', 'Eval_Freq'] 
#dirs = ['Eval_size', 'Eval_Total']
mode = 0

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

for dir in dirs:
    folders = os.listdir(dir)
    folders = [x for x in folders if not x.startswith('_') and not x.startswith('.') and x.endswith('.txt')]
    folders.sort()

    os.chdir(dir)

    tuples = []
    data = []
    node_configs = set()

    for x in folders:
        split = x.split('.')
        split = split[0].split('_')
        num_nodes = int(split[-1])
        name = '_'.join(split[:-1])
        tuples.append([num_nodes, name])
        node_configs.add(num_nodes)
        with open(x) as f:
            lines = [line.rstrip() for line in f][-num_values:]
            lines = [line for line in lines if not line.startswith('[')]
            results = [float(line.split(' ')[-1]) for line in lines]
            if len(results) < num_values:
                diff = num_values - len(results)
                results += [np.mean(results)] * diff
            data.append(results)

    data = np.array(data).T
    index = pd.MultiIndex.from_tuples(tuples, names=["nodes", "name"])
    df = pd.DataFrame(data, columns=index)

    for x in node_configs:
        # Convert multi-index DataFrame to a regular one for plotting
        df_plot = df[x].reset_index()
        df_plot_melted = df_plot.melt(id_vars='index', var_name='Benchmark', value_name='Runtime')

        # Create a box plot with Plotly
        fig = px.box(df_plot_melted, x='Benchmark', y='Runtime', title=f'Benchmark Comparison for {x} Nodes')
        fig.update_layout(xaxis_title="Benchmark", yaxis_title="Runtime in seconds", xaxis_tickangle=-70)
        fig.write_image(f'comparison_{x}.pdf')  # Save as PDF

    weakscale = ['Benchmark_results_GPU', 'Benchmark_results_ANY', 'Benchmark_results_GPU_multi', 'Benchmark_results_ANY_multi', 'Reference_GPU_results', 'Reference_GPU_Dataopt_results']
    if mode == 0:
        weakscale = ['Benchmark_results_GPU', 'Benchmark_results_ANY']
    

    scaling_data, err_data = [], []

    if mode == 0:
        references = ['Reference_GPU_results', 'Reference_GPU_Dataopt_results']
        base = 44
        for ref in references:
            # copy the reference data to the other node configurations
            for n in node_configs:
                df[n, ref] = df[base, ref]


    for leg in weakscale:
        # Filter outliers in df[x] using the IQR method
        for x in node_configs:
            Q1 = df[x][leg].quantile(0.25)
            Q3 = df[x][leg].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df.loc[df[x][leg].index, (x, leg)] = df[x][leg][(df[x][leg] >= lower_bound) & (df[x][leg] <= upper_bound)]
        scaling_data.append([df[x].mean()[leg] for x in node_configs])
        err_data.append([df[x].std()[leg] for x in node_configs])

    if mode == 0:
        print("Weak scaling:")
        for leg in ['Benchmark_results_GPU', 'Benchmark_results_ANY']:
            print(f"  {leg} {2}: {df[1][leg].mean() / df[2][leg].mean()}")
            print(f"  {leg} {44}: {df[1][leg].mean() / df[44][leg].mean()}")
            print(f"  {leg} {44}_2: {df[2][leg].mean() / df[44][leg].mean()}")
            print(f"  {leg} 1: {df[1][leg].mean()}")
            print(f"  {leg} 2: {df[2][leg].mean()}")
            print(f"  {leg} 44: {df[44][leg].mean()}")

    scaling_data = np.array(scaling_data).T
    err_data = np.array(err_data).T

    weakscale = ['TARGETDART_OFFLOAD', 'TARGETDART_ANY', 'TARGETDART_OFFLOAD multi', 'TARGETDART_ANY multi', 'OMP target', 'OMP target w/o data-transfer']
    if mode == 0:
        weakscale = ['TARGETDART_OFFLOAD', 'TARGETDART_ANY']

    if mode == 0 or mode == 2:
        node_configs = [i * 4 for i in node_configs]

    node_configs = list(node_configs)

    if mode == 3:
        df2 = pd.DataFrame(scaling_data, columns=weakscale, index=node_configs).sort_index(ascending=False)
        df2_err = pd.DataFrame(err_data, columns=weakscale, index=node_configs).sort_index(ascending=False)
    else:
        df2 = pd.DataFrame(scaling_data, columns=weakscale, index=node_configs).sort_index()
        df2_err = pd.DataFrame(err_data, columns=weakscale, index=node_configs).sort_index()

    if mode == 1:
        df2 = df2.drop(100)
        for col in df2.columns:
            first_value = df2[col].iloc[0]
            last_value = df2[col].iloc[-1]
            ratio = last_value / first_value if first_value != 0 else None
            print(f"Column: {col}, Ratio (last/first): {ratio} Last value: {last_value} First value: {first_value}")

    if mode == 2:
        avg_over_configs = df2.mean(axis=0)
        print("Average over all configurations for all columns:")
        print(avg_over_configs)
    
    if mode == 3:
        for col in df2.columns:
            first_value = df2[col].iloc[0]
            last_value = df2[col].iloc[-1]
            ratio = last_value / first_value if first_value != 0 else None
            print(f"Column: {col}, Ratio (last/first): {ratio} Last value: {last_value} First value: {first_value}")

    # Create a line plot with error bars using Plotly
    fig = go.Figure()

    if mode == 3:
        node_configs = sorted(node_configs, reverse=True)
    else:
        node_configs = sorted(node_configs)

    minimode = 0
    for col in df2.columns:
        fig.add_trace(go.Scatter(
            x=df2.index,
            y=df2[col],
            mode='lines+markers',
            line=dict(dash='solid' if minimode == 0 else 'dash' if minimode == 1 else 'dot' if minimode == 2 else 'dashdot'),
            name=col,
            error_y=dict(type='data', array=df2_err[col], visible=True),
            marker=dict(
                symbol="diamond" if minimode == 0 else "square" if minimode == 1 else "circle" if minimode == 2 else "cross",
                size=10
            )
        ))
        minimode = minimode + 1
    
    if mode == 0:
        # Add a horizontal line at the value of the first measurement for all columns
        for col in df2.columns:
            first_value = df2[col].iloc[0]
            fig.add_hline(
                y=first_value,
                line_dash="dot",
                line_color="gray",
            )

    xname = 'Load Shift in %' if mode == 1 else 'Number of GPUs' if mode == 0 else 'Number of GPUs' if mode == 2 else 'Frequency in MHz'

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=xname,
        yaxis_title="Runtime in seconds",
        template='plotly_white',
        yaxis_rangemode='tozero',  # Ensure y-axis starts from 0
        xaxis=dict(
            tickmode='array',
            tickvals=[i * 16 for i in range(1, (max(node_configs) // 16) + 1)]
        ) if mode == 0 else dict(
            tickmode='array',
            tickvals=[i * 4 for i in range(1, (max(node_configs) // 4) + 1)]
        ) if mode == 2 else dict(autorange='reversed') if mode == 3 else dict(autorange=True),  # Use multiples of 20 for x-axis if mode is 0
        legend=dict(
            x=0.05,
            y=0.05,
            xanchor='left',
            yanchor='bottom'
        ) if mode == 0 or mode == 2 else dict(
            x=0.05,
            y=0.65,
            xanchor='left',
            yanchor='bottom'
        )
    )

    #fig.show()

    if mode == 0:
        print("Weak scaling:")
        for col in df2.columns:
            print(f"  avgs: {df2[col].mean()}")

    name = 'weakscaling' 
    if mode == 1:
        name = 'scaling_migration'
    elif mode == 2:
        name = 'overhead'
    elif mode == 3:
        name = 'frequency'

    fig.write_image(f'{name}.pdf', format='pdf')
    #fig.write_image('weakscaling.jpg' if mode == 0 else 'scaling_migration.jpg', format='jpg')

    os.replace(f'{name}.pdf', f'../{name}.pdf')

    os.chdir("..")
    mode += 1

    #fig.show()
