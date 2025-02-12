import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the text
with open('presentation.txt', 'r') as f:
    text = f.read()

vocab = sorted(set(text))  # Get all unique characters
vocab_size = len(vocab) + 1  # Number of unique characters + 1 to account for 1-based indexing
vocab2index = {c: i+1 for i, c in enumerate(vocab)}  # Character to index mapping, starting from 1
index2vocab = {i+1: c for i, c in enumerate(vocab)}  # Index to character mapping, starting from 1
context_size = 8 # Number of characters to consider as context
batch_size = 32 # Number of samples per batch
epoch_size = 50 # Number of times to iterate over the entire dataset

N = 1024 # Number of samples in the signal
t = torch.linspace(0, 1, N, requires_grad=False) # Time axis for the signal

def get_first_n_primes(n):
    """Generate first n prime numbers"""
    primes = []
    num = 2
    while len(primes) < n:
        if all(num % prime != 0 for prime in primes):
            primes.append(num)
        num += 1
    return primes

# After vocab initialization, create prime number mapping
prime_numbers = get_first_n_primes(vocab_size)
char_to_prime = {i: prime for i, prime in enumerate(prime_numbers, start=1)}  # 1-based indexing

def context_to_signal(context):
    signal = torch.zeros(N)
    for i, item in enumerate(context):
        # Use prime numbers for frequencies
        freq = char_to_prime[item]
        signal += torch.sign(torch.sin(2 * torch.pi * freq * (t + i/N)))
    return signal / (torch.max(torch.abs(signal)) + 1e-12)

def sample_from_output(output, temperature=1.0):
    if temperature < 1e-3:
        return torch.argmax(output).item()

    # Adjust output with temperature
    output = output / temperature
    # Compute probabilities using softmax on the correct dimension
    probabilities = F.softmax(output, dim=0)  # Use dim=0 for 1D tensor

    # Sample from the probability distribution
    return torch.multinomial(probabilities, num_samples=1).item()


# Function to generate the dataset
def generate_dataset(text, context_size):
    X = []
    Y = []
    for i in range(len(text)):
        if i < context_size:
            X.append(text[:i])
        else:
            X.append(text[i-context_size:i])
        Y.append(text[i])
    return X, Y

# Generate the dataset
X, Y = generate_dataset(text, context_size)


print(f"Number of samples: {len(X)}, {len(Y)}")

class TextDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    
    def __getitem__(self, index):
        signal = context_to_signal([vocab2index[c] for c in self.X[index]])
        return signal, torch.tensor([vocab2index[self.Y[index]]], dtype=torch.long)
    
    def __len__(self):
        return len(self.X)


dataset = TextDataset(X, Y)

# Optimize dataloader for hardware
num_workers = 4 if device.type == "cuda" else 0
pin_memory = device.type == "cuda"
dataloader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory
)

class MyModel(nn.Module):
    def __init__(self, N, dropout_rate=0.2):
        super(MyModel, self).__init__()
        
        self.DECIDE = nn.Sequential(
            nn.Linear(N, N),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(N, vocab_size)  # vocab_size is now +1 larger to account for 1-based indexing
        )

    def forward(self, input_signal):
        return self.DECIDE(input_signal)


# Move model to device and optimize for hardware
model = MyModel(N, dropout_rate=0.2)
model = model.to(device)

if device.type == "cuda":
    model = torch.compile(model)  # Uses torch dynamo to optimize model
    
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Add this function before the training loop
def generate_text(model, start_text, length=32, temperature=0.9):
    model.eval()
    with torch.no_grad():
        current_text = start_text
        for _ in range(length):
            context = current_text[-context_size:]
            encoded = [vocab2index[c] for c in context]
            signal = context_to_signal(encoded).to(device)
            output = model(signal.unsqueeze(0))
            pred = sample_from_output(output[0], temperature)
            current_text += index2vocab[pred]
    return current_text

# Modify the training loop
print("\nStarting training...\n")

for epoch in range(epoch_size):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epoch_size}")
    
    for i, (x, y) in enumerate(progress_bar):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=device.type=="cuda"):
            output = model(x)
            loss = criterion(output, y.squeeze())
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{total_loss/(i+1):.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
    
    # Save checkpoint and generate sample text every 10 epochs
    if (epoch + 1) % 5 == 0:
        # Save model checkpoint
        checkpoint_path = f'model_checkpoint_epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"\nSaved checkpoint to {checkpoint_path}")
        
        # Generate sample text
        print("\nGenerating sample text:")
        sample_text = generate_text(model, "My name ", length=64)
        print(f"Generated text: {sample_text}\n")
        model.train()  # Set back to training mode

print("\nTraining completed!")

# Modify the generation loop to use the generate_text function
with torch.no_grad():
    model.eval()
    while True:
        sentence = input("\nEnter a starting text (or 'quit' to exit): ")
        if sentence.lower() == 'quit':
            break
        generated = generate_text(model, sentence, length=500)
        print(f"\nGenerated text:\n{generated}\n")
