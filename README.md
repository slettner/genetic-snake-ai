# Genetic Snake
Neural networks are nowadays commonly trained with gradient based methods. 
Here, I take a look into an alternative. A genetic algorithm directly optimizes the weight of the 
neural network. The goal is, to learn playing the game of snake. 
In addition, I tried using only binary weights for the neural network. 
The GA is able to teach the snakes decent strategies achieving scores 
up 50 apples on a 15x15 field in about 2 hours of searching.
I am not the first who did this, there are several other successful snakes trained by a genetic algorithm
[1](https://becominghuman.ai/designing-ai-solving-snake-with-evolution-f3dd6a9da867),
[2](https://theailearner.com/2018/11/09/snake-game-with-genetic-algorithm/) and I got most of my ideas from there.


## Genetic Algorithm 
There is enough literature out there explaining the basic principle of a Genetic Algorithm,
so it won't be repeated here. [This](https://towardsdatascience.com/introduction-to-genetic-algorithms-including-example-code-e396e98d8bf3)
blog post gives a nice introduction. The following only explains the operators used in the  
successful runs. 

### Selection
In the selection phase I look at the parents and the children of the current generation
and *greedily* pick the best n snakes from all of them, where 
n is set to the initial population size. This has elitism inherent and keeps the population 
at the same size. It could be an improvement to also include some solutions worse than the best 
n to support diversity but I did not investigate that.

### Crossover
The crossover is implemented by simple *random-k-point-crossover*.
It is implemented such that two parent chromosomes produce two child chromosomes during crossover.
First, I select k-location in the chromosome. Then I then iterate over the 
k points and flip the chromosome content between the current and the next k every second step.
The new arrays are the children and the old one are kept as the parents.

### Mutation
For the mutation operator I used *random gaussian noise* in case of the neural network with continuous 
weights. In case of the binary neural network the mutation operator randomly flips bits. 
The mutation is applied to each child of a generation with a probability of 30%. 
Also, not every value of the chromosome is affected by the mutation but rather a random subset of 
n values, where n is below 10% of the chromosomes total size.


### Fitness
The fitness function is simple. To evaluate it the snake plays one game. 
For each collected apple the fitness is increased by one. In addition, each step the snake 
survives the fitness is incremented by 0.001. This prefers snakes which achieve not running into 
walls. The snake has a maximum of 225 steps to reach the current apple. This makes sure that a snake which reached 
one apple is always better than on which just stays alive. In addition, there needs to be a terminal criterion to 
address endless-loop snakes. 


## Neural Network 

Both, the normal and the binary neural network consist of *fully-connected* layers. 
The best score was achieved with a single hidden layer containing 16 neurons for the neural 
network with continuous weights. *Relu* was used as the activation and the initial weights
were sampled from a *random normal distribution*.

## State
The state is what the snake sees at every step it makes. I.e. its the input to the neural network
which outputs the action the snake will take in this step.
I tried two different state representations. The one which worked better contained seven values:
* left blocked (0 or 1)
* front blocked (0 or 1)
* right blocked (0 or 1)
* snake direction in x (-1 or 1)
* snake direction in y (-1 or 1)
* apple angle relative to snake (degree)
* apple angle relative to snake (degree)

The representation which worked less well (the snake did not learn at all) 
had 24 values and was based on distances to wall, snake and apple.


## Action
At each step the snake can select between going left, right or continue straight. 
I.e. the neural network has three output neurons where each represents one action.
Using softmax activations on the output we select the action with highest probability.



## Results
[![](http://img.youtube.com/vi/nLp6u8bRUaA/0.jpg)](http://www.youtube.com/watch?v=nLp6u8bRUaA "Snake AI")

The video shows a snake reaching a score above 40 apple. 
In order to run your own snake install the requirements
```
>> pip install -r requirements.txt
>> pip install .
>> pip install external/genetic-alogirhtm/src
```
To run the training script, execute
```
>> python3 experiments/snake-evolution/train_snakes.py
```

The gin configuration framework is used to set all parameters. In the folder 'experiments/snake-evolution' is
a file called 'evolution.gin' which configures the training. 
If you want to save snakes change the 'snake_dir' from None to your directory.


## Docker

The code dockerized and can be build in 'docker/' by executing
```
>> bash make.sh
```


## Conclusion 
Although the genetic algorithm is less prone to get stuck in a local optimum
the scalability is 'questionable' as the search space explodes for large network architecture. 
The GA was not able to train the binary neural network to produce snake with acceptable performance. 
Further hyper-parameter tuning might help thought. 

