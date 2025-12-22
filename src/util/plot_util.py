import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torch


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

def plot_timesteps(episode_durations, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
    plt.ioff()
    plt.show()

def plot_simple(data: list, x_label="episode", y_label="reward") -> None: 
    min_value = torch.min(data).item()
    max_value = torch.max(data).item()
    mean_value = round(torch.mean(data.float()).item(), 2)
    print("shape:", data.shape)
    title = "Results mean=" + str(mean_value) + ", min=" + str(min_value) + ", max=" + str(max_value)
    print("title:", title)
    plt.figure(figsize=(10, 5))
    plt.plot(data)
    
    # Set labels and title
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Display the plot
    #plt.show()
    plt.savefig("results/mtorch-rewards.png")   



import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def plot_with_torch(data: torch.Tensor, x_label="episode", y_label="reward") -> None: 
    # Ensure data is a float tensor for calculations
    data_float = data.float()
    
    min_value = torch.min(data).item()
    max_value = torch.max(data).item()
    mean_value = round(torch.mean(data_float).item(), 2)
    
    plt.figure(figsize=(10, 5))
    
    # 1. Plot the raw data (faded in the background)
    plt.plot(data.cpu(), alpha=0.3, label="Raw " + y_label)
    
    # 2. Calculate the 50-moving average
    # We use padding to keep the output the same size as the input
    window_size = 50
    if len(data) >= window_size:
        # Reshape to (Batch, Channels, Length) for avg_pool1d
        # We use reflect padding to avoid "zero-drop" at the edges
        padded_data = data_float.view(1, 1, -1)
        moving_avg = F.avg_pool1d(padded_data, kernel_size=window_size, stride=1, padding=window_size//2)
        moving_avg = moving_avg.flatten()[:len(data)] # Trim to match original length
        
        plt.plot(moving_avg.cpu(), color='red', linewidth=2, label=f"{window_size} Avg")

    # Set labels and title
    title = f"Results mean={mean_value}, min={min_value}, max={max_value}"
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("results/mtorch-rewards.png")
    plt.close() # Good practice to close figure to free memory    