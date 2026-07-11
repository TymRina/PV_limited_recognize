import pandas as pd
import plotly.graph_objects as go
import sys

def plot_csv(file_path):
    df = pd.read_csv(file_path)
    
    if df.empty:
        print("Error: CSV file is empty")
        return
    
    time_col = df.columns[0]
    value_cols = df.columns[1:]
    
    fig = go.Figure()
    
    for col in value_cols:
        fig.add_trace(go.Scatter(
            x=df[time_col],
            y=df[col],
            mode='lines',
            name=col,
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         f'{time_col}: %{{x}}<br>' +
                         'Value: %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'Time Series Plot - {file_path}',
        xaxis_title=time_col,
        yaxis_title='Value',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1d', step='day', stepmode='backward'),
                dict(count=7, label='1w', step='day', stepmode='backward'),
                dict(count=1, label='1m', step='month', stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(visible=True),
        type='date' if pd.api.types.is_datetime64_any_dtype(df[time_col]) else 'category'
    )
    
    html_file = file_path.replace('.csv', '.html')
    fig.write_html(html_file)
    print(f"Plot saved to {html_file}")
    
    fig.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_csv.py <csv_file_path>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    plot_csv(csv_file)