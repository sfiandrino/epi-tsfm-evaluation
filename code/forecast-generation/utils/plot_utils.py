import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates


def plot_forecast(pred_df, surv_data, suptitle=None, start_date='2024-10-18', fig_format="US", color="#4ECED4", location_names_dict=None, show_legend=True, savefig=False, save_path=None):
    """Plot Forecasts"""

    # cast datetime
    surv_data.truth_date = pd.to_datetime(surv_data.truth_date)
    pred_df.target_end_date = pd.to_datetime(pred_df.target_end_date)

    # ref_date and location list
    ref_date = pred_df.reference_date.values[0]
    locations = sorted(pred_df.location.unique())

    #add 'US' at the beginning
    if 'US' in locations:
        locations = ['US'] + [s for s in locations if s != "US"]
    else: 
        locations = np.sort(locations)

    # figure panels format
    if fig_format == "US": 
        fig, axes = plt.subplots(13, 5, figsize=(20, 2.75 * 13))
    else: 
        fig, axes = plt.subplots(6, 4, figsize=(20, 2.75 * 7))
    
    if suptitle: 
        fig.suptitle("Forecasts as of %s)" % ref_date, fontsize=22, y=1.)
    else: 
        fig.suptitle(suptitle, fontsize=22, y=1.)

    # plot
    for location, ax in zip(locations, axes.flatten()):

        # plot surveillance
        df = surv_data[surv_data.location == location]
        df = df[(df.truth_date >= start_date) & (df.truth_date <= pred_df.target_end_date.max())]
        ax.plot(df.truth_date.values, df["value"].values, "ko-", label="Data")

        # plot forecast
        df_forecast = pred_df[pred_df.location == location]
        df_forecast['output_type_id'] = np.round(df_forecast['output_type_id'], 3)

        if df_forecast.shape[0]!=0:
            meds = df_forecast[df_forecast['output_type_id'] == 0.5].sort_values(by = "target_end_date")
            lower_iqr = df_forecast[df_forecast['output_type_id'] == 0.25].sort_values(by = "target_end_date")
            upper_iqr = df_forecast[df_forecast['output_type_id'] == 0.75].sort_values(by = "target_end_date")
            lower_95 = df_forecast[df_forecast['output_type_id'] == 0.05].sort_values(by = "target_end_date")
            upper_95 = df_forecast[df_forecast['output_type_id'] == 0.95].sort_values(by = "target_end_date")

            ax.plot(meds.target_end_date.values, meds['value'].values, "o-", color=color, label="Median")
            ax.fill_between(lower_iqr.target_end_date.values, lower_iqr['value'].values, upper_iqr['value'].values, alpha=0.25, color=color, label="IQR", linewidth=0)
            ax.fill_between(lower_95.target_end_date.values, lower_95['value'].values, upper_95['value'].values, alpha=0.2, color=color, label="90% CI", linewidth=0)

        else:
            ax.set_facecolor("#dcdcdc")
            ax.text(0.3, 0.8, "No Forecast", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=13)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
        
        if location_names_dict is None: 
            ax.set_title(location, fontsize = 18)
        else: 
            ax.set_title(location_names_dict[location], fontsize = 18)

    # show legend
    if show_legend: 
        axes[0,0].legend()
    
    # hide remaining axes
    for ax in axes.flatten()[pred_df.location.nunique():]:
        ax.axis('off')

    plt.tight_layout()

    # save figure
    if savefig:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
