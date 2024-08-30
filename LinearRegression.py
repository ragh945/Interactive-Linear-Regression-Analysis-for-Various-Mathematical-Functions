import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import streamlit as st
from PIL import Image

# Display an image at the top of the app
math=Image.open("Math.png")
Inno = Image.open("Inno.png")
st.image(Inno,use_column_width=True)
st.image(math)
# Add custom CSS for dark overlay background
st.markdown(
    """
    <style>
    body {
        margin: 0;
        padding: 0;
        background: url('https://wallpapercave.com/wp/wp8154260.jpg') no-repeat center center fixed; 
        background-size: cover;
    }
    .reportview-container {
        background: rgba(0, 0, 0, 0.5); /* Dark overlay with 50% opacity */
        position: relative;
        min-height: 100vh;
    }
    .sidebar .sidebar-content {
        background-color: rgba(255, 255, 255, 0.8); /* Optional: Semi-transparent background for the sidebar */
    }
    .element-container {
        background-color: rgba(255, 255, 255, 0.9); /* Optional: White background for content area */
    }
    </style>
    """,
    unsafe_allow_html=True
)




# Functions for generating data
def create_straight_line(n_points, slope, intercept, noise_level):
    x = np.linspace(-10, 10, n_points)
    y = slope * x + intercept + np.random.normal(0, noise_level, x.shape)
    return x, y

def create_parabola(n_points, a, b, c, noise_level):
    x = np.linspace(-10, 10, n_points)
    y = a * x**2 + b * x + c + np.random.normal(0, noise_level, x.shape)
    return x, y

def create_hyperbola(n_points, a, b, noise_level):
    x = np.linspace(-10, 10, n_points)
    y = a / (x - b) + np.random.normal(0, noise_level, x.shape)
    return x, y

def create_pair_of_straight_lines(n_points, slope1, intercept1, slope2, intercept2, noise_level):
    x1 = np.linspace(-10, 0, n_points // 2)
    y1 = slope1 * x1 + intercept1 + np.random.normal(0, noise_level, x1.shape)
    
    x2 = np.linspace(0, 10, n_points // 2)
    y2 = slope2 * x2 + intercept2 + np.random.normal(0, noise_level, x2.shape)
    
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    return x, y

def create_cubic(n_points, noise_level):
    x = np.linspace(-10, 10, n_points)
    y = x**3 - 3*x**2 + 2*x + np.random.normal(0, noise_level, x.shape)
    return x, y

def create_quartic(n_points, noise_level):
    x = np.linspace(-10, 10, n_points)
    y = x**4 - 4*x**3 + 6*x**2 - 4*x + 1 + np.random.normal(0, noise_level, x.shape)
    return x, y

def create_quintic(n_points, noise_level):
    x = np.linspace(-10, 10, n_points)
    y = x**5 - 5*x**4 + 10*x**3 - 10*x**2 + 5*x - 1 + np.random.normal(0, noise_level, x.shape)
    return x, y

def create_modx(n_points, noise_level):
    x = np.linspace(-10, 10, n_points)
    y = np.abs(x) + np.random.normal(0, noise_level, x.shape)
    return x, y

def create_expx(n_points, noise_level):
    x = np.linspace(-2, 2, n_points)
    y = np.exp(x) + np.random.normal(0, noise_level, x.shape)
    return x, y

def create_logx(n_points, noise_level):
    x = np.linspace(0.1, 10, n_points)  # Avoid log(0) by starting from a small positive number
    y = np.log(x) + np.random.normal(0, noise_level, x.shape)
    return x, y

def linear_regression(x, y):
    x = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    return y_pred

# Streamlit app
st.title("Interactive Linear Regression Analysis for Various Mathematical Functions")

# Sidebar for function selection
st.sidebar.title("Select Function")
function_type = st.sidebar.selectbox("Function Type", 
                                     ["Straight Line", "Parabola", "Hyperbola", 
                                      "Pair of Straight Lines", "Cubic Function", 
                                      "Quartic Function", "Quintic Function", 
                                      "mod(x)", "exp(x)", "log(x)"])

# Sidebar for additional parameters
n_points = st.sidebar.slider("Number of Data Points", min_value=50, max_value=500, value=100, step=50)
noise_level = st.sidebar.slider("Noise Level", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

# Generate data based on function type
if function_type == "Straight Line":
    slope = st.sidebar.slider("Slope", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
    intercept = st.sidebar.slider("Intercept", min_value=-10.0, max_value=10.0, value=3.0, step=0.1)
    x, y = create_straight_line(n_points, slope, intercept, noise_level)
elif function_type == "Parabola":
    a = st.sidebar.slider("Coefficient a", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
    b = st.sidebar.slider("Coefficient b", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
    c = st.sidebar.slider("Coefficient c", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
    x, y = create_parabola(n_points, a, b, c, noise_level)
elif function_type == "Hyperbola":
    a = st.sidebar.slider("Coefficient a", min_value=1.0, max_value=10.0, value=1.0, step=0.1)
    b = st.sidebar.slider("Coefficient b (Horizontal Shift)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
    x, y = create_hyperbola(n_points, a, b, noise_level)
elif function_type == "Pair of Straight Lines":
    slope1 = st.sidebar.slider("Slope 1", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
    intercept1 = st.sidebar.slider("Intercept 1", min_value=-10.0, max_value=10.0, value=3.0, step=0.1)
    slope2 = st.sidebar.slider("Slope 2", min_value=-10.0, max_value=10.0, value=-2.0, step=0.1)
    intercept2 = st.sidebar.slider("Intercept 2", min_value=-10.0, max_value=10.0, value=-3.0, step=0.1)
    x, y = create_pair_of_straight_lines(n_points, slope1, intercept1, slope2, intercept2, noise_level)
elif function_type == "Cubic Function":
    x, y = create_cubic(n_points, noise_level)
elif function_type == "Quartic Function":
    x, y = create_quartic(n_points, noise_level)
elif function_type == "Quintic Function":
    x, y = create_quintic(n_points, noise_level)
elif function_type == "mod(x)":
    x, y = create_modx(n_points, noise_level)
elif function_type == "exp(x)":
    x, y = create_expx(n_points, noise_level)
elif function_type == "log(x)":
    x, y = create_logx(n_points, noise_level)

# Perform linear regression and plot
y_pred = linear_regression(x, y)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Actual Points')
plt.plot(x, y_pred, color='red', label='Linear Regression Line')
plt.grid()
plt.title(f"{function_type} with Linear Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
st.pyplot(plt)
