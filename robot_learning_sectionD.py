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

        self.network = None


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
    def create_optimal_seq(self, num_action_sequences, num_actions_per_sequence, environment,initial_state=np.array([0.5, 0.5])):

        min_action=self.robot.min_action
        max_action=self.robot.max_action


        means=[[(min_action+max_action)/2,(min_action+max_action)/2] for _ in range(num_actions_per_sequence)]
        stds=[[max_action,max_action] for _ in range(num_actions_per_sequence)]

        means_per_step=[]

        for time in range(5): #refit P n times

            seqs = []
            rewards = []

            for nb_seq in range(num_action_sequences):
                seq_actions=[]


                seq_positions=[initial_state]

                reward=0

                for nb_act in range(num_actions_per_sequence):
                    act_1=np.random.normal(means[nb_act][0],stds[nb_act][0])
                    act_2 = np.random.normal(means[nb_act][1],stds[nb_act][1])
                    print("actions")

                    act_1 = np.clip(act_1, min_action, max_action)
                    act_2 = np.clip(act_2, min_action, max_action)

                    act=np.array([act_1,act_2])
                    print(act)
                    seq_actions.append(act)


                    #self.robot.take_action(act,environment)



                    input=torch.tensor(np.array([seq_positions[-1][0], seq_positions[-1][1], act_1, act_2])).float()
                    print(input)
                    print(time)
                    print(nb_seq)
                    print(nb_act)
                    final_position=np.array([self.network.forward(input)[0].item(),self.network.forward(input)[1].item()])
                    print("final_pos")
                    print(final_position)
                    goal_position=environment.goal_state

                    seq_positions.append(final_position)




                    if goal_position.tolist()==final_position.tolist():
                        reward+=np.inf

                    else:
                        distance=np.sqrt((final_position[0]-goal_position[0])**2+(final_position[1]-goal_position[1])**2)
                        reward+=1/distance

                rewards.append(reward)

                seqs.append(seq_actions)


            #we now have n sequences and each of them have a specific reward

            means=[]
            stds=[]

            index=np.flip(np.argsort(rewards))

            #we keep the 5% best sequence

            bests_index=index[:int(len(index)*0.005)]
            print("bests_index")
            print(bests_index)

            bests_seqs=[seqs[idx] for idx in bests_index]

            for i in range(num_actions_per_sequence):
                actions_i_0=[]
                actions_i_1 = []
                for seq_i in bests_seqs:
                    actions_i_0.append(seq_i[i][0])
                    actions_i_1.append(seq_i[i][1])

                means.append([np.mean(actions_i_0),np.mean(actions_i_1)])
                stds.append([np.std(actions_i_0),np.std(actions_i_1)])

            means_per_step.append(means)



        seqs = np.array(seqs)
        best_seq = seqs[np.argmax(rewards)]

        self.policy = best_seq
        #print(rewards)
        #print(best_seq)

        return best_seq






    # Function to calculate a policy using random shooting planning
    def calculate_policy_with_random_shooting(self, num_action_sequences, num_actions_per_sequence, environment):

        self.network = self.create_neural_net(num_action_sequences, num_actions_per_sequence, environment)


        position=np.array([0.5,0.5])
        list_positions=[position]


        for i in range(num_actions_per_sequence):

            seq=self.create_optimal_seq(num_action_sequences,num_actions_per_sequence,environment,position)
            if i==0:

                initial_seq=seq

            action=seq[0]

            position=self.network.forward(torch.tensor(np.array([position[0], position[1], action[0], action[1]])).float())

            list_positions.append(position)

        return initial_seq,list_positions


    def create_dataset(self, num_action_sequences, num_actions_per_sequence, environment):

        min_action=self.robot.min_action*100
        max_action=self.robot.max_action*100

        dataset_moves=[]

        for nb_seq in range(num_action_sequences):
            seq_actions=[]


            for nb_act in range(num_actions_per_sequence):
                old_position = self.robot.state

                act_1=np.random.randint(min_action,max_action)/100
                act_2 = np.random.randint(min_action, max_action)/100


                act=np.array([act_1,act_2])
                seq_actions.append(act)

                self.robot.take_action(act,environment)
                new_position = self.robot.state
                dataset_moves.append([old_position, act, new_position])

        return dataset_moves


    def create_neural_net(self,num_action_sequences, num_actions_per_sequence, environment):
        dataset_moves=self.create_dataset(num_action_sequences, num_actions_per_sequence, environment)

        random.shuffle(dataset_moves)

        nb_data = len(dataset_moves)

        train_set = np.array(dataset_moves[:int(0.8 * nb_data)])

        test_set = np.array(dataset_moves[int(0.8 * nb_data):])



        train_input_data = np.array([np.array([elt[0][0], elt[0][1], elt[1][0], elt[1][1]]) for elt in train_set])
        train_label_data = np.array([np.array([elt[2][0], elt[2][1]]) for elt in train_set])

        test_input_data = np.array([np.array([elt[0][0], elt[0][1], elt[1][0], elt[1][1]]) for elt in test_set])
        test_label_data = np.array([np.array([elt[2][0], elt[2][1]]) for elt in test_set])



        test_input_tensor = torch.tensor(test_input_data).float()
        test_label_tensor = torch.tensor(test_label_data).float()
        nb_tests = len(test_input_data)

        # Create the neural network
        network = Network(input_dimension=4, output_dimension=2)
        # Create the optimiser
        optimiser = torch.optim.Adam(network.parameters(), lr=0.001)

        # Create lists to store the losses and epochs
        losses = []
        iterations = []
        test_losses = []

        # Create a graph which will show the loss as a function of the number of training iterations
        fig, ax = plt.subplots()
        ax.set(xlabel='Iteration', ylabel='Loss', title='Loss Curve for Torch Example')

        # Loop over training iterations
        for training_iteration in range(10):
            # Set all the gradients stored in the optimiser to zero.
            optimiser.zero_grad()
            # Sample a mini-batch of size 5 from the training data

            # print(int(len(train_input_data)/100))
            minibatch_indices = np.random.choice(range(len(train_input_data)), int(len(train_input_data) / 100))


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


        return network



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
        [self.initial_seq, self.list_positions]=self.agent.calculate_policy_with_random_shooting(num_action_sequences=1000, num_actions_per_sequence=30,
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



        #then we do  by the first one in red
        positions = []

        for position in self.initial_seq:

            pos=[settings.SCREEN_SIZE * position[0],settings.SCREEN_SIZE * position[1]]
            positions.append(pos)
            arcade.draw_circle_filled(pos[0],pos[1],radius=5, color=[255,102,102])

        arcade.draw_line_strip(point_list=positions,color=[255,102,102],line_width=3)

        #we do the last one
        positions = []

        for position in self.list_positions:

            pos=[settings.SCREEN_SIZE * position[0],settings.SCREEN_SIZE * position[1]]
            positions.append(pos)
            arcade.draw_circle_filled(pos[0],pos[1],radius=5, color=[0,255,0])

        arcade.draw_line_strip(point_list=positions,color=[0,255,0],line_width=3)


        pass


# The main entry point
if __name__ == "__main__":

    # Create a new program, which will also do the robot's initial planning
    MainProgram()

    # Run the main Arcade loop forever
    # This will repeatedly call the MainProgram.on_update() and MainProgram.on_draw() functions.
    arcade.run()






