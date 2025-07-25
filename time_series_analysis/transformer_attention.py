import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Generate Synthetic Time Series Data ---
# Create a series with a strong seasonal pattern (e.g., weekly) so attention has something to learn.
time = np.arange(0, 400)
# A strong seasonal component every 28 days + some trend + noise
seasonality_period = 28
amplitude = 10
trend_slope = 0.05
noise_level = 1.5
series = (
    amplitude * np.sin(2 * np.pi * time / seasonality_period)
    + trend_slope * time
    + np.random.randn(len(time)) * noise_level
)


# --- 2. Prepare Data for the Model ---
def create_inout_sequences(input_data, tw):
    """Creates sliding window sequences for time series forecasting."""
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i : i + tw]
        train_label = input_data[i + tw : i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


input_window = 56  # Use a window of two full cycles (2 * 28)
all_sequences = create_inout_sequences(series, input_window)

# Convert to PyTorch tensors
# Using np.array first is more efficient as the warning suggests
X_np = np.array([seq for seq, _ in all_sequences])
y_np = np.array([label for _, label in all_sequences])
X = torch.FloatTensor(X_np).unsqueeze(-1)  # Add feature dim
y = torch.FloatTensor(y_np)


# --- 3. Define the Transformer Model (Same as before) ---
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=32, nhead=4, num_layers=1, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear_net = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.input_projection(src)
        attn_output, attn_weights = self.attention(src, src, src, need_weights=True)
        src = self.norm1(src + self.dropout(attn_output))
        ff_output = self.linear_net(src)
        src = self.norm2(src + self.dropout(ff_output))
        prediction_input = src[:, -1, :]
        output = self.output_layer(prediction_input)
        return output, attn_weights


# --- 4. Train the Model (Briefly - Same as before) ---
model = TimeSeriesTransformer(nhead=4, d_model=32)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 30
# Combine data for easier batching if desired, but simple loop is fine for this example
train_data = torch.utils.data.TensorDataset(X, y)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

for epoch in range(epochs):
    for seq, labels in train_loader:
        optimizer.zero_grad()
        model.train()
        y_pred, _ = model(seq)
        loss = criterion(y_pred, labels)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

# --- 5. Visualize Attention for a Single Sample (CORRECTED LOGIC) ---
model.eval()

# Let's pick a sample from our dataset
sample_idx = 150
input_sample = X[sample_idx].unsqueeze(0)  # Add batch dimension

# Get prediction and attention weights
prediction, attention_weights = model(input_sample)

# attention_weights shape is [batch_size, query_len, key_len] -> [1, 56, 56]
# The weights are already averaged across heads by default.

# **THE CRITICAL FIX IS HERE:**
# We don't need the faulty .mean() line. We slice directly from the output.
# We want the attention scores from the LAST query step to all key steps.
last_step_attention = attention_weights[:, -1, :]  # -> Shape [1, 56]

# Convert to a NumPy array for plotting.
attention_to_plot = last_step_attention.detach().numpy()

# Create the plot
fig, ax = plt.subplots(figsize=(12, 2.5))
sns.heatmap(attention_to_plot, cmap="viridis", cbar=True, ax=ax)

ax.set_xlabel("Past Time Steps (Input History)", fontsize=12)
ax.set_yticklabels([])
ax.set_ylabel("Prediction\nStep", rotation=0, labelpad=40, va="center", fontsize=12)
ax.set_title(
    "Transformer Attention: Focus on Past Data for a Single Forecast",
    fontsize=14,
    pad=20,
)
plt.tight_layout()
plt.show()
