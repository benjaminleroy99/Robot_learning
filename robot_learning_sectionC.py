# Imports
import arcade
import numpy as np

import settings
import environment
import robot
import random

from torch_example import *

dataset_moves=[]


# The Agent class, which represents the robot's "brain"
class Agent:

    def __init__(self):

        # Create a robot, which represents the physical robot in the environment (the agent is just the "brain")
        self.robot = robot.Robot()
        # Initialise a policy to empty, which will later be assigned to
        self.policy = None
        # Set the policy's action index to zero
        self.policy_index = 0



    # Function to take a physical action in the physical environment
    def take_action(self, environment):

        # Get the next action from the policy
        if self.policy_index>=30:
            self.policy_index=0

        next_action = self.policy[self.policy_index]

        # Execute this action and hence update the state of the robot
        self.robot.take_action(next_action, environment)

        # Increment the index of the policy's next action
        self.policy_index += 1

        #print(self.policy_index)

    # Function to calculate a policy using random shooting planning
    def calculate_policy_with_random_shooting(self, num_action_sequences, num_actions_per_sequence, environment):

        min_action=self.robot.min_action*100
        max_action=self.robot.max_action*100

        seqs=[]
        rewards=[]

        for nb_seq in range(num_action_sequences):
            seq_actions=[]


            for nb_act in range(num_actions_per_sequence):
                old_position = self.robot.state
                act_1=np.random.randint(min_action,max_action)/100
                act_2 = np.random.randint(min_action, max_action) / 100
                act=np.array([act_1,act_2])
                seq_actions.append(act)
                #environment.dynamics(self.robot.state,act)
                self.robot.take_action(act,environment)
                new_position = self.robot.state
                dataset_moves.append([old_position, act, new_position])



            final_position=self.robot.state
            goal_position=environment.goal_state

            if goal_position.tolist()==final_position.tolist():
                reward=np.inf
            else:
                distance=np.sqrt((final_position[0]-goal_position[0])**2+(final_position[1]-goal_position[1])**2)
                reward=1/distance

            rewards.append(reward)

            seqs.append(seq_actions)
            self.robot.state = np.array([0.5, 0.5])


        seqs=np.array(seqs)

        best_seq=seqs[np.argmax(rewards)]

        self.policy=best_seq
        #print(rewards)
        #print(best_seq)

        return self.neural_net()

    def neural_net(self):

        random.shuffle(dataset_moves)

        nb_data = len(dataset_moves)

        train_set = np.array(dataset_moves[:int(0.8 * nb_data)])

        test_set = np.array(dataset_moves[int(0.8 * nb_data):])

        '''
        print("heya")
        print(len(train_set))
        print(train_set)
        print(train_set[0][1])
        '''

        train_input_data = np.array([np.array([elt[0][0], elt[0][1], elt[1][0], elt[1][1]]) for elt in train_set])
        train_label_data = np.array([np.array([elt[2][0], elt[2][1]]) for elt in train_set])

        test_input_data = np.array([np.array([elt[0][0], elt[0][1], elt[1][0], elt[1][1]]) for elt in test_set])
        test_label_data = np.array([np.array([elt[2][0], elt[2][1]]) for elt in test_set])

        '''
        print(len(train_label_data))
        print(train_label_data)

        print(len(test_label_data))
        print(test_label_data)
        '''

        test_input_tensor = torch.tensor(test_input_data).float()
        test_label_tensor = torch.tensor(test_label_data).float()
        nb_tests = len(test_input_data)

        # Create the neural network
        network = Network(input_dimension=4, output_dimension=2)
        # Create the optimiser
        optimiser = torch.optim.Adam(network.parameters(), lr=0.01)

        # Create lists to store the losses and epochs
        losses = []
        iterations = []
        test_losses = []

        # Create a graph which will show the loss as a function of the number of training iterations
        fig, ax = plt.subplots()
        ax.set(xlabel='Iteration', ylabel='Loss', title='Loss Curve for Torch Example')

        # Loop over training iterations
        for training_iteration in range(750):
            # Set all the gradients stored in the optimiser to zero.
            optimiser.zero_grad()
            # Sample a mini-batch of size 5 from the training data

            # print(int(len(train_input_data)/100))
            minibatch_indices = np.random.choice(range(len(train_input_data)), int(len(train_input_data) /100))
            minibatch_inputs = train_input_data[minibatch_indices]
            minibatch_labels = train_label_data[minibatch_indices]

            # Convert the NumPy array into a Torch tensor
            minibatch_input_tensor = torch.tensor(minibatch_inputs).float()
            minibatch_labels_tensor = torch.tensor(minibatch_labels).float()
            # Do a forward pass of the network using the inputs batch
            network_prediction = network.forward(minibatch_input_tensor)
            # Compute the loss based on the label's batch
            '''
            print("prediction")
            print(network_prediction)
            print("labels")
            print(minibatch_labels_tensor)
            '''

            loss = torch.nn.MSELoss()(network_prediction, minibatch_labels_tensor)
            # Compute the gradients based on this loss,
            # i.e. the gradients of the loss with respect to the network parameters.
            loss.backward()
            # Take one gradient step to update the network
            optimiser.step()
            # Get the loss as a scalar value
            loss_value = loss.item()
            # Print out this loss
            print('Iteration ' + str(training_iteration) + ', Loss = ' + str(loss_value))
            # Store this loss in the list
            losses.append(loss_value)
            # Update the list of iterations
            iterations.append(training_iteration)
            # Plot and save the loss vs iterations graph

            test_loss = 0

            for test_i in range(nb_tests):
                test_network_prediction = network.forward(test_input_tensor[test_i])

                test_loss += torch.nn.MSELoss()(test_network_prediction, test_label_tensor[test_i])

            test_loss = test_loss / nb_tests

            print(test_loss)

            test_losses.append(test_loss.item())

        ax.plot(iterations, losses, color='blue')
        ax.plot(iterations, test_losses, color='red')
        plt.yscale('log')
        plt.show()
        fig.savefig("loss_vs_iterations.png")

        exe_action = np.array([0.05, 0.05])

        init_states = [np.array([0.1, 0.1]), np.array([0.1, 0.9]), np.array([0.9, 0.1]), np.array([0.9, 0.9]),
                       np.array([0.5, 0.5])]

        inputs = []
        for state in init_states:
            inputs.append(torch.tensor(np.array([state[0], state[1], exe_action[0], exe_action[1]])).float())

        print(inputs)

        preds = []
        for input in inputs:
            pred = network.forward(input)
            preds.append(pred)

        #preds=np.array(preds)
        print(preds)

        return inputs,preds


# The main Program class
class MainProgram(arcade.Window):

    # Initialisation function to create a new program
    def __init__(self):
        super().__init__(width=settings.SCREEN_SIZE, height=settings.SCREEN_SIZE, title=settings.SCREEN_TITLE, update_rate=1.0/settings.UPDATE_RATE)

        # Create the environment
        self.environment = environment.Environment()

        # Create the agent
        self.agent = Agent()

        # Set the environment's background colour
        arcade.set_background_color(settings.BACKGROUND_COLOR)

        # Initialise the time step to the beginning of time
        self.time_step = 0

        # Do random shooting planning
        [self.inputs,self.preds]=self.agent.calculate_policy_with_random_shooting(num_action_sequences=1000, num_actions_per_sequence=30,
                                                         environment=self.environment)

    # on_update is called once per loop and is used to update the robot / environment
    def on_update(self, delta_time=30):

        # On each timestep, the agent will execute the next action in its policy
        # This is the policy that was already calculated using planning in the function
        while self.time_step<delta_time:
            self.agent.take_action(self.environment)
            # Update the time step
            self.time_step += 1

    # on_draw is called once per loop and is used to draw the environment
    def on_draw(self):

        # Clear the screen
        arcade.start_render()

        # Draw the environment
        self.environment.draw()

        '''
        # Draw the robot
        self.agent.robot.draw()

        
        position = self.agent.robot.state

        for action in self.agent.policy:
            new_position=position+action

            if new_position[0] > self.environment.obstacle_state[0] - 0.5 * self.environment.obstacle_size[0]:
                if new_position[0] < self.environment.obstacle_state[0] + 0.5 * self.environment.obstacle_size[0]:
                    if new_position[1] > self.environment.obstacle_state[1] - 0.5 * self.environment.obstacle_size[1]:
                        if new_position[1] < self.environment.obstacle_state[1] + 0.5 * self.environment.obstacle_size[1]:
                            # The robot's next state is inside the obstacle,
                            # so set the robot's next state to its current state
                            new_position = position


            arcade.draw_circle_filled(settings.SCREEN_SIZE * new_position[0], settings.SCREEN_SIZE * new_position[1], settings.ROBOT_SIZE, settings.ROBOT_COLOUR)
            # You may want to add code here to draw the policy, the sampled paths, or any other visualisations
            position=new_position
        '''



        nb_pred=len(self.preds)

        couple=[]

        for i in range(nb_pred):

            pos1=[settings.SCREEN_SIZE * self.inputs[i][0].item(),settings.SCREEN_SIZE * self.inputs[i][1].item()]

            pos2=[settings.SCREEN_SIZE * self.preds[i][0].item(),settings.SCREEN_SIZE * self.preds[i][1].item()]


            arcade.draw_circle_filled(pos1[0],pos1[1],radius=5, color=[0,255,0])
            arcade.draw_circle_filled(pos2[0],pos2[1],radius=5, color=[0,0,255])

            couple.append([pos1,pos2])



        for c in couple:
            arcade.draw_line_strip(point_list=c,color=[180,180,180])



        pass


# The main entry point
if __name__ == "__main__":

    # Create a new program, which will also do the robot's initial planning
    MainProgram()

    # Run the main Arcade loop forever
    # This will repeatedly call the MainProgram.on_update() and MainProgram.on_draw() functions.
    arcade.run()






