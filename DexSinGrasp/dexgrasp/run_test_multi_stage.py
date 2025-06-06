import subprocess
import re

total_successes = 0
total_steps = 0
num_trials = 100
non_zero_steps = 0

for i in range(num_trials):
    # Run singulation.py with current seed
    result = subprocess.run(['python', 'singulation.py', '--seed', str(i)], 
                          capture_output=True, text=True)
    
    # Get output as string
    output = result.stdout
    
    # Extract success/failure
    if 'tensor([1.], device=\'cuda:0\')' in output:
        total_successes += 1
    print(f"Success: {total_successes} out of {i+1}")
    # Write progress to file
    with open('progress.txt', 'a') as f:
        f.write(f"Success: {total_successes} out of {i+1}\n")
        
    # Extract steps number from output
    step_match = re.search(r'steps (\d+)', output)
    print(step_match)
    # Write step match to file
    with open('progress.txt', 'a') as f:
        f.write(f"Step match: {step_match}\n")
        
    if step_match:
        current_steps = float(step_match.group(1))
        if current_steps > 0:
            total_steps += current_steps
            non_zero_steps += 1

# Calculate final statistics
success_rate = total_successes / num_trials
avg_steps = total_steps / non_zero_steps if non_zero_steps > 0 else 0

print(f"\nFinal Results:")
print(f"Success Rate: {success_rate:.2%}")
print(f"Average Steps: {avg_steps:.2f}")

# Write final results to file
with open('progress.txt', 'a') as f:
    f.write("\nFinal Results:\n")
    f.write(f"Success Rate: {success_rate:.2%}\n")
    f.write(f"Average Steps: {avg_steps:.2f}\n")