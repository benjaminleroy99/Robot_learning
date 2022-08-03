# Imports
import arcade
import numpy as np

import settings
import environment
import robot

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
        if self.policy_index==len(self.policy):
            self.policy_index=0
        next_action = self.policy[self.policy_index]
        # Execute this action and hence update the state of the robot
        self.robot.take_action(next_action, environment)
        # Increment the index of the policy's next action
        self.policy_index += 1

    # Function to calculate a policy using random shooting planning
    def calculate_policy_with_random_shooting(self, num_action_sequences, num_actions_per_sequence, environment):

        min_action=self.robot.min_action
        max_action=self.robot.max_action



        means=[[(min_action+max_action)/2,(min_action+max_action)/2] for _ in range(num_actions_per_sequence)]
        means_x = [(min_action+max_action)/2 for _ in range(num_actions_per_sequence)]
        means_y = [(min_action+max_action)/2 for _ in range(num_actions_per_sequence)]
        covs=[max_action*np.eye(num_actions_per_sequence),max_action*np.eye(num_actions_per_sequence)]



        means_per_step=[]

        for time in range(5): #refit P n times

            seqs = []
            rewards = []


            for nb_seq in range(num_action_sequences):


                seq_actions=[]

                seq_actions_x=np.random.multivariate_normal(means_x, covs[0])
                seq_actions_y=np.random.multivariate_normal(means_y, covs[1])

                self.robot.state = np.array([0.5, 0.5])

                seq_positions=[self.robot.state]

                reward=0

                for nb_act in range(num_actions_per_sequence):
                    action=np.array([seq_actions_x[nb_act],seq_actions_y[nb_act]])

                    action[0] = np.clip(action[0], min_action, max_action)
                    action[1] = np.clip(action[1], min_action, max_action)


                    seq_actions.append(action)
                    self.robot.take_action(action,environment)
                    seq_positions.append(self.robot.state)


                    final_position=self.robot.state
                    goal_position=environment.goal_state

                    if goal_position.tolist()==final_position.tolist():
                        reward+=np.inf
                    else:
                        distance=np.sqrt((final_position[0]-goal_position[0])**2+(final_position[1]-goal_position[1])**2)
                        reward+=1/distance

                rewards.append(reward)

                seqs.append(seq_actions)



            means=[]
            means_x = []
            means_y = []

            values_x=[]
            values_y=[]



            index=np.flip(np.argsort(rewards))

            #we keep the 5% best sequence

            bests_index=index[:int(len(index)*0.005)]


            bests_seqs=[seqs[idx] for idx in bests_index]

            for i in range(num_actions_per_sequence):
                actions_i_x=[]
                actions_i_y = []
                for seq_i in bests_seqs:
                    actions_i_x.append(seq_i[i][0])
                    actions_i_y.append(seq_i[i][1])


                means.append([np.mean(actions_i_x),np.mean(actions_i_y)])
                means_x.append(np.mean(actions_i_x))
                means_y.append(np.mean(actions_i_y))

                values_x.append(actions_i_x)
                values_y.append(actions_i_y)


            covs = [np.cov(values_x),np.cov(values_y)]




            means_per_step.append(means)

        seqs = np.array(seqs)
        best_seq = seqs[np.argmax(rewards)]

        self.policy = best_seq
        #print(rewards)
        #print(best_seq)

        return means_per_step


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
        self.means_per_step=self.agent.calculate_policy_with_random_shooting(num_action_sequences=1000, num_actions_per_sequence=30,
                                                         environment=self.environment)

        print("verif les means")
        print(self.means_per_step)

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

        # Draw the robot
        #self.agent.robot.draw()

        initial_position=np.array([0.5,0.5])

        sequences=[]

        nb_it=len(self.means_per_step)
        nb_actions=len(self.means_per_step[0])

        for i in range(nb_it):
            positions_i=[initial_position]

            for action_i in range(nb_actions):
                positions_i.append(positions_i[-1]+self.means_per_step[i][action_i])

            sequences.append(positions_i)



        #all appart from the first and last
        for i in range(1,len(sequences)-1):
            seqq=sequences[i]
            positions = []
            for position in seqq:
                pos=[settings.SCREEN_SIZE * position[0],settings.SCREEN_SIZE * position[1]]
                positions.append(pos)
                arcade.draw_circle_filled(pos[0],pos[1],radius=5, color=[180,180,180])
            arcade.draw_line_strip(point_list=positions,color=[180,180,180],line_width=3)

        #then we do  by the first one in red
        positions = []

        for position in sequences[0]:

            pos=[settings.SCREEN_SIZE * position[0],settings.SCREEN_SIZE * position[1]]
            positions.append(pos)
            arcade.draw_circle_filled(pos[0],pos[1],radius=5, color=[255,102,102])

        arcade.draw_line_strip(point_list=positions,color=[255,102,102],line_width=3)

        #we do the last one
        positions = []

        for position in sequences[-1]:

            pos=[settings.SCREEN_SIZE * position[0],settings.SCREEN_SIZE * position[1]]
            positions.append(pos)
            arcade.draw_circle_filled(pos[0],pos[1],radius=5, color=[0,255,0])

        arcade.draw_line_strip(point_list=positions,color=[0,255,0],line_width=3)



        arcade.draw_circle_filled(settings.SCREEN_SIZE * self.environment.goal_state[0],
                                  settings.SCREEN_SIZE * self.environment.goal_state[1], settings.SCREEN_SIZE * settings.GOAL_SIZE,
                                  settings.GOAL_COLOUR)

# The main entry point
if __name__ == "__main__":

    # Create a new program, which will also do the robot's initial planning
    MainProgram()

    # Run the main Arcade loop forever
    # This will repeatedly call the MainProgram.on_update() and MainProgram.on_draw() functions.
    arcade.run()
