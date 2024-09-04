#Author: Abdallah Ikbarieh
#Date: September 3, 2024
import streamlit as st
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(6, 1000)
        self.fc2 = nn.Linear(1000, 750)
        self.fc3 = nn.Linear(750, 300)
        self.fc4 = nn.Linear(300, 5)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)

        Reg1 = F.softplus(x[:,0:1], beta=10, threshold=20)
        Reg2 = F.softplus(x[:,1:2], beta=10, threshold=20)
        Reg3 = x[:,2:3]
        Reg4 = x[:,3:4]
        Reg5 = x[:,4:5]
        x = torch.cat([Reg1, Reg2, Reg3, Reg4, Reg5], dim=1)

        return x
        
model = NeuralNet()
model.load_state_dict(torch.load('neural_network_model.pth'))
model.eval()


# Load scalers
with open('scalers.pkl', 'rb') as f:
    scaler_data = pickle.load(f)

# Load scalers
with open('scalers.pkl', 'rb') as f:
    scaler_data = pickle.load(f)

# Streamlit interface
def main():
    st.title("Machine Learning-based Hopper Design for Flowing Woody Biomass")

    # Display the image
    st.image("App_ML.png", width=500)

    # Inputs
    #ps = st.number_input("Enter d_50 in mm:", format="%.2f")
    st.markdown("### Enter mean particle size d<sub>50</sub> in mm:", unsafe_allow_html=True)
    ps = st.number_input("ps", key="ps", label_visibility="collapsed", value=2.00)
    st.markdown("### Enter wetbased moisture content MC% in percentage:")
    mc = st.number_input("mc", key="mc", label_visibility="collapsed", value=20.00)
    st.markdown("### Enter relative density D<sub>r</sub> using numeric values:", unsafe_allow_html=True)
    st.markdown("##### D<sub>r</sub>= 0: very loose; D<sub>r</sub>= 1: loose; D<sub>r</sub>= 2: Dense", unsafe_allow_html=True)
    rd = st.selectbox("rd", [0, 1, 2], key="dr", label_visibility="collapsed", index=1)
    st.markdown("### Enter wall friction coefficient &mu;<sub>w</sub> in fractions:", unsafe_allow_html=True)
    wf = st.number_input("wf", key="wf", label_visibility="collapsed", value=0.250)
    st.markdown("### Enter hopper inclination angle &Theta; in degrees:", unsafe_allow_html=True)
    ia = st.number_input("ia", key="ia", label_visibility="collapsed", value=30.00)
    st.markdown("### Enter hopper opening width W in mm:")
    w = st.number_input("w", key="w", label_visibility="collapsed", value=75.00)
    st.markdown("### Enter hopper out-of-domain dimension in m:")
    L = st.number_input("L", key="L", label_visibility="collapsed", value=1.000, format="%.3f")

    # Button to make predictions
    if st.button('Predict'):
        # Collect all input values
        inputs = np.array([ps, mc, rd, wf, ia, w])

        # Apply scaling only to the inputs that require it
        scaled_inputs = np.array([
            (ps - scaler_data['scaler_X_mean'][0]) / scaler_data['scaler_X_scale'][0],  # Scaling ps
            (mc/100 - scaler_data['scaler_X_mean'][1]) / scaler_data['scaler_X_scale'][1],  # Scaling mc
            rd,  # No scaling for rd
            (wf - scaler_data['scaler_X_mean'][2]) / scaler_data['scaler_X_scale'][2],  # Scaling wf
            (ia - scaler_data['scaler_X_mean'][3]) / scaler_data['scaler_X_scale'][3],  # Scaling ia
            (w - scaler_data['scaler_X_mean'][4]) / scaler_data['scaler_X_scale'][4]    # Scaling w
        ])

        # Convert the scaled inputs into a tensor for prediction
        e = torch.tensor(scaled_inputs, dtype=torch.float32).unsqueeze(0)
        
        # Predict output using the model
        pred_output = model(e)

        # Process and display the output
        output_preprocessing(pred_output, L)



def output_preprocessing(pred_output, L):
    # Output scaling
    MFR = (pred_output[0,0]*(scaler_data['scaler_y_max'][0] - scaler_data['scaler_y_min'][0]) + scaler_data['scaler_y_min'][0])/0.4*L
    Is = pred_output[0,1]*(scaler_data['scaler_y_max'][1] - scaler_data['scaler_y_min'][1]) + scaler_data['scaler_y_min'][1]
    C1 = pred_output[0,2].item()*(scaler_data['scaler_y_max'][2] - scaler_data['scaler_y_min'][2]) + scaler_data['scaler_y_min'][2]
    C2 = pred_output[0,3].item()*(scaler_data['scaler_y_max'][3] - scaler_data['scaler_y_min'][3]) + scaler_data['scaler_y_min'][3]
    C3 = pred_output[0,4].item()*(scaler_data['scaler_y_max'][4] - scaler_data['scaler_y_min'][4]) + scaler_data['scaler_y_min'][4]

    st.write(f'Average mass flow rate (MFR)= {MFR:.3f} tonne/hr')
    st.write(f'Smoothness index (Is)= {Is:.3f}')
    st.write(f'C1= {C1:.3f}')
    st.write(f'C2= {C2:.3f}')
    st.write(f'C3= {C3:.3f}')

    # Plotting
    x_values = np.linspace(-1, 1, 21)
    y_values = x_values**2 * C1 + x_values * C2 + C3
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values)
    ax.set_title("Flow Pattern")
    ax.set_xlabel("Normalized x [-]")
    ax.set_ylabel("Normalized Velocity [-]")
    ax.set_ylim(0, 1.0)
    ax.grid(True)
    st.pyplot(fig)

if __name__ == '__main__':
    main()
