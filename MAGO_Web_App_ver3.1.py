import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import logging
import datetime

# Create a file to log the hydrograph plotting process.
def create_log(logfile):
    # Remove all existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=logfile,
        filemode='w'  # overwrite each run
    )

# Base polynomial
def go_poly(hw, a, b, c):
    return a*hw**2 + b*hw + c

def no_mago_poly(tw, a, b, c, d, e):
    return a*tw**4 + b*tw**3 + c*tw**2 + d*tw + e
    
def interpolate_value_vec(hw_point, tw_point, curves):
    # Compute TW values for all curves at this HW
    tw_curves = np.array([c["func"](hw_point) for c in curves])
    values = np.array([c["value"] for c in curves])
    
    # Clamp if outside
    if tw_point <= tw_curves.min():
        return values.min()
    if tw_point >= tw_curves.max():
        return values.max()
    
    # Find which interval the TW lies in
    idx = np.where((tw_curves[:-1] <= tw_point) & (tw_point <= tw_curves[1:]))[0][0]
    tw1, tw2 = tw_curves[idx], tw_curves[idx+1]
    v1, v2 = values[idx], values[idx+1]
    
    # Linear interpolation
    return v1 + (tw_point - tw1)/(tw2 - tw1)*(v2 - v1)

def interpolate_value_clamped(tw_point, hw_point, curves, no_mago_line):
    if no_mago_line:
        # Compute TW values for all No MAGO curves at this HW
        hw_curve = np.array([c["func"](tw_point) for c in no_mago_line])
        # print(tw_point, hw_curve)        
        # Point is to the right if TW > max(TW_curve)
        if hw_point < hw_curve.max():
            return "No MAGO"
        else:
            # Get y-values of all curves at this x
            tw_curve = [c["func"](hw_point) for c in curves]
            values   = [c["value"] for c in curves]
            
            # Below the first curve → clamp to first value
            if tw_point < tw_curve[0]:
                return f"<{values[0]} ft"
            
            # Above the last curve → clamp to last value
            if tw_point > tw_curve[-1]:
                return values[-1]
            
            # Interpolate between the two nearest curves
            for i in range(len(curves)-1):
                tw1, tw2 = tw_curve[i], tw_curve[i+1]
                v1, v2 = values[i], values[i+1]
                if tw1 <= tw_point <= tw2:
                    return v1 + (tw_point - tw1)/(tw2 - tw1) * (v2 - v1)
            
            # Fallback (should not happen)
            return np.nan
   
    else:
        # Get y-values of all curves at this x
        tw_curve = [c["func"](hw_point) for c in curves]
        values   = [c["value"] for c in curves]
        
        # Below the first curve → clamp to first value
        if tw_point < tw_curve[0]:
            return f"<{values[0]} ft"
        
        # Above the last curve → clamp to last value
        if tw_point > tw_curve[-1]:
            return values[-1]
        
        # Interpolate between the two nearest curves
        for i in range(len(curves)-1):
            tw1, tw2 = tw_curve[i], tw_curve[i+1]
            v1, v2 = values[i], values[i+1]
            if tw1 <= tw_point <= tw2:
                return v1 + (tw_point - tw1)/(tw2 - tw1) * (v2 - v1)
        
        # Fallback (should not happen)
        return np.nan



def main():
    try:
        # Define work directory - Determine the correct base path for accessing files
        if getattr(sys, 'frozen', False):
            workdir = sys._MEIPASS  # PyInstaller temp directory
        else:
            workdir = os.path.dirname(os.path.abspath(__file__))
        # workdir = r"Q:\Tools_Apps\MAGO_Checks\MAGO_Curve_Check_Automation"        
        
        # Initiate log file
        logfile = os.path.join(workdir, "mago_web_app.log")       
        create_log(logfile)
        
        # List of stations for MAGO calculation
        stationnames = os.path.join(workdir, "structureList.csv")
        df_sta = pd.read_csv(stationnames)
       
        df_go_coeff = pd.read_csv(os.path.join(workdir, 'poly_fit_coeff.csv'))
        logging.info(df_go_coeff)
        
        df_no_mago_coeff = pd.read_csv(os.path.join(workdir, 'poly_no_mago_coeff.csv'))
        logging.info(df_no_mago_coeff)

        df_domain = pd.read_csv(os.path.join(workdir, 'domain.csv'))
        logging.info(df_domain)
        
        df_dailydata = pd.read_csv(os.path.join(workdir, 'daily_data.csv'))
        logging.info(df_dailydata)
        
        st.sidebar.header("Select Structure")
        structurelist = df_sta['Structure'].unique()
        selected_structure = st.sidebar.selectbox("Choose a structure:", structurelist, key='structure')
        

        # Get daily data for the selected structure
        filtered_dailydata = df_dailydata[df_dailydata['Structure'] == selected_structure]
        daily_hw = filtered_dailydata['Headwater (ft-NAVD88)'].iloc[0]
        daily_tw = filtered_dailydata['Tailwater (ft-NAVD88)'].iloc[0]
        
        # --- Initialize session state ---
        if "last_structure" not in st.session_state:
            st.session_state.last_structure = selected_structure
        
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = "file_uploader"
        
        if "hw_point" not in st.session_state:
            st.session_state.hw_point = daily_hw
        
        if "tw_point" not in st.session_state:
            st.session_state.tw_point = daily_tw
        
        # --- Reset function ---
        def current_conditions():
            st.session_state.hw_point = daily_hw
            st.session_state.tw_point = daily_tw
        
        def reset_inputs():
            st.session_state.hw_point = daily_hw
            st.session_state.tw_point = daily_tw
            st.session_state.uploader_key = str(datetime.datetime.now())
            st.session_state.last_structure = selected_structure
        
        # --- Auto reset when structure changes ---
        if selected_structure != st.session_state.last_structure:
            st.session_state.update({
                "hw_point": daily_hw,
                "tw_point": daily_tw,
                "uploader_key": str(datetime.datetime.now()),
                "last_structure": selected_structure
            })
            st.rerun()        
       
        
        # --- Filter datasets ---
        filtered_go_coeff = df_go_coeff[df_go_coeff['Structure'] == selected_structure]
        filtered_no_mago_coeff = df_no_mago_coeff[df_no_mago_coeff['Structure'] == selected_structure]        
        filtered_df_domain = df_domain[df_domain['Structure'] == selected_structure]
        
        HW_max = filtered_df_domain['HW_max_NAVD88'].iloc[0]
        HW_min = filtered_df_domain['HW_min_NAVD88'].iloc[0]
        TW_max = filtered_df_domain['TW_max_NAVD88'].iloc[0]
        TW_min = filtered_df_domain['TW_min_NAVD88'].iloc[0]
        
        # --- Sidebar inputs ---
        #st.sidebar.header("Enter HW & TW Conditions")
        st.sidebar.button("Current Conditions", on_click=current_conditions)
        
        tw_point = st.sidebar.number_input(
            "Tailwater Level (TW)", step=0.01, key="tw_point"
        )
        hw_point = st.sidebar.number_input(
            "Headwater Level (HW)", step=0.01, key="hw_point"
        )
        
        # --- File uploader (with dynamic key) ---
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file with HW & TW",
            type=["csv"],
            key=st.session_state.uploader_key
        )
        
        # --- Manual Reset Button ---
        st.sidebar.button("Reset Inputs & File", on_click=reset_inputs)

        # Create curves   
        curves = []
        for idx, row in filtered_go_coeff.iterrows():
            a, b, c = row['a'], row['b'], row['c']
            go_value = row['GO_ft']
            curves.append({
                "func": lambda hw, a=a, b=b, c=c: go_poly(hw, a, b, c),
                "value": go_value
            })
        
        no_mago_line = []
        for idx, row in filtered_no_mago_coeff.iterrows():
            a, b, c, d, e = row['a'], row['b'], row['c'], row['d'], row['e']
            go_value = row['GO_ft']
            no_mago_line.append({
                "func": lambda tw, a=a, b=b, c=c, d=d, e=e: no_mago_poly(tw, a, b, c, d, e),
                "value": go_value
            })
           
        hw = np.linspace(HW_min, HW_max, 400)
        tw = np.linspace(TW_min, TW_max, 400)
        
        fig, ax = plt.subplots(figsize=(16,12))
        
        # Plot No MAGO curve(s)
        for c in no_mago_line:
            x_vals = tw
            y_vals = c["func"](tw)
            mask = ((y_vals >= HW_min) & (y_vals <= HW_max) &
                    (x_vals >= TW_min) & (x_vals <= TW_max))
            x_trim = x_vals[mask]
            y_trim = y_vals[mask]
            
            ax.plot(x_trim, y_trim, color="black", linewidth=2, label = 'No MAGO Restriction Curve')
      
        if no_mago_line:
            no_mago_curve = no_mago_line[0]["func"](tw)
            
        prev_x_trim = None
        prev_y_trim = None    
        
        for idx, c in enumerate(curves):
            y_vals = hw                      # HW domain
            x_vals = c["func"](hw)           # TW values for this curve
            
            if no_mago_line:
                # Interpolate No MAGO HW at these TW values
                no_mago_y = np.interp(x_vals, tw, no_mago_curve)
        
                # Mask within domain ranges
                mask = (
                    (y_vals >= HW_min) & (y_vals <= HW_max) &
                    (x_vals >= TW_min) & (x_vals <= TW_max)
                )
        
                # Difference: positive means curve above No MAGO
                d = y_vals - no_mago_y
        
                # Find indices where sign changes => intersections
                sign_changes = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
        
                if len(sign_changes) >= 2:
                    # Take the *upper* intersection (largest y)
                    i2 = sign_changes[-1]
        
                    # Linear interpolation to get exact intersection y (h2)
                    y1, y2 = y_vals[i2], y_vals[i2+1]
                    d1, d2 = d[i2], d[i2+1]
                    h2 = y1 - d1 * (y2 - y1) / (d2 - d1)  # interpolated HW at intersection
        
                    # Keep only points above this intersection
                    mask = mask & (y_vals >= h2)
        
                else:
                    # Otherwise keep normal "above No MAGO"
                    mask = mask & (d >= 0)
        
                x_trim = x_vals[mask]
                y_trim = y_vals[mask]
                
                if prev_x_trim is not None and prev_y_trim is not None:                             
                    prev_y_interp = np.interp(x_trim, prev_x_trim, prev_y_trim)
                    if selected_structure == 'S65EX1':
                        d = y_trim - prev_y_interp                 
                        mask2 = (d <= 0)
                        x_trim = x_trim[mask2]
                        y_trim = y_trim[mask2]                   
                    
        
                # Save trimmed for next iteration
                prev_x_trim, prev_y_trim = x_vals, y_vals
                
            else:
                mask = (
                    (y_vals >= HW_min) & (y_vals <= HW_max) &
                    (x_vals >= TW_min) & (x_vals <= TW_max)
                )
                x_trim = x_vals[mask]
                y_trim = y_vals[mask]

                 
            if len(x_trim) > 0:
                if idx == 0:
                    ax.plot(x_trim, y_trim, linestyle = '--', color = 'k', label = 'MAGO Curves')
                else:
                    ax.plot(x_trim, y_trim, linestyle = '--', color = 'k')
        
                # Annotate once, in the middle of the trimmed portion
                mid_idx = len(x_trim)//2
                # mid_idx = np.argmax(y_trim)
                xm, ym = x_trim[mid_idx], y_trim[mid_idx]
                dx = x_trim[min(mid_idx+1, len(x_trim)-1)] - x_trim[max(mid_idx-1, 0)]
                dy = y_trim[min(mid_idx+1, len(y_trim)-1)] - y_trim[max(mid_idx-1, 0)]
                angle = np.degrees(np.arctan2(dy, dx))
                
                ax.text(xm, ym - 0.01, f'{c["value"]} ft',   # ↑ adjust offset as needed
                         rotation=angle, rotation_mode='anchor',
                         fontsize=12, color='blue',
                         ha='center', va='bottom', clip_on=True)
        
        # Test point
        val_clamped = interpolate_value_clamped(tw_point, hw_point, curves, no_mago_line)
        ax.scatter(tw_point, hw_point, c='red', s=80, zorder=5)
        
        if isinstance(val_clamped, str):
            annotation = val_clamped
        elif not np.isnan(val_clamped):
            annotation = f"{val_clamped:.2f} ft"
        else:
            annotation = "Outside curves"
        
        ax.annotate(annotation, (tw_point, hw_point),
                     textcoords="offset points", xytext=(10,10), fontsize = 12,
                     color="red", clip_on=True)
        
        
        ax.set_xlabel("TW Elevation (feet NAVD88)", fontsize = 12)
        ax.set_ylabel("HW Elevation (feet NAVD88)", fontsize = 12)
        ax.set_title(f"Polynomial-Fitted MAGO Curves for {selected_structure}", fontsize = 16)
        ax.grid(True)
        # ax.set_xlim(TW_min, TW_max)
        # ax.set_ylim(HW_min, HW_max)
        if tw_point < TW_min:    
            ax.set_xlim(tw_point - 0.5, TW_max)
        elif tw_point > TW_max:
            ax.set_xlim(TW_min, tw_point + 0.5)
        else:
            ax.set_xlim(TW_min, TW_max)
        
        if hw_point < HW_min:    
            ax.set_ylim(hw_point - 0.5, HW_max)
        elif hw_point > HW_max:
            ax.set_ylim(HW_min, hw_point + 0.5)
        else:
            ax.set_ylim(HW_min, HW_max)
        
        ax.legend(fontsize='large', title='Legend', title_fontsize='x-large', loc='lower right')
        
        # Set title of the app
        st.title("Polynomial-Fitted MAGO Calculator")
        
        # Add styled subheader
        if isinstance(annotation, str):
            st.markdown(
                f"""
                <h3 style="font-size:20px; text-align:center; color:#2c3e50;">
                    Calculated Maximum Allowable Gate Opening (MAGO) for {selected_structure}: {annotation}
                </h3>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <h3 style="font-size:20px; text-align:center; color:#2c3e50;">
                    Calculated Maximum Allowable Gate Opening (MAGO) for {selected_structure}: {round(annotation, 1)} feet
                </h3>
                """,
                unsafe_allow_html=True
            )
        
        
        if uploaded_file is not None:
            try:
                input_data = pd.read_csv(uploaded_file)
        
                if {'HW_NAVD', 'TW_NAVD'}.issubset(input_data.columns):
                    # Compute MAGO values for all points
                    results = []
                    for idx, row in input_data.iterrows():
                        hw = row['HW_NAVD']
                        tw = row['TW_NAVD']
                        predicted_mago = interpolate_value_clamped(tw, hw, curves, no_mago_line)
        
                        if isinstance(predicted_mago, str):
                            mago = predicted_mago
                        else:
                            mago = round(predicted_mago, 1)
        
                        results.append({
                            "Structure": selected_structure,
                            "HW_NAVD88": hw,
                            "TW_NAVD88": tw,
                            "MAGO (ft)": mago
                        })
        
                    df_output = pd.DataFrame(results)
        
                    # Plot ALL uploaded points at once (instead of inside loop)
                    ax.scatter(
                        input_data['TW_NAVD'], 
                        input_data['HW_NAVD'], 
                        c='red', s=80, zorder=5, marker = 'x', label="Uploaded CSV Points"
                    )
                    
                    for xi, yi, zi in zip(input_data['TW_NAVD'], input_data['HW_NAVD'], df_output["MAGO (ft)"]):
                        ax.text(xi, yi+0.02, f'{zi}', fontsize=13, ha='center', va='bottom', color = 'gray')
        
                    # Ensure only one legend entry for uploaded points
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys(), fontsize='large', title='Legends', title_fontsize='x-large', loc='lower right')
        
                    # Prepare downloadable output
                    dt = datetime.datetime.now()
                    formatted_dt = dt.strftime("%Y-%m-%d_%H:%M:%S")
                    output_file = f"{selected_structure}_HW_TW_MAGO_{formatted_dt}.csv"
                    st.download_button(
                        label="Download Processed Data",
                        data=df_output.to_csv(index=False),
                        file_name=output_file,
                        mime="text/csv"
                    )
                    st.success("Batch processing completed successfully!")
                else:
                    st.error("Uploaded CSV must contain 'HW_NAVD' and 'TW_NAVD' columns.")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        # Plot the graph
        st.pyplot(fig)
        

    except Exception as e:
        logging.error(f"This is an error: {e}")
        raise

      
if __name__ == "__main__":
    main()







