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
        next_action = self.policy[self.policy_index]
        # Execute this action and hence update the state of the robot

        self.robot.take_action(next_action, environment)

        # Increment the index of the policy's next action
        self.policy_index += 1

    # Function to calculate a policy using random shooting planning
    def calculate_policy_with_random_shooting(self, num_action_sequences, num_actions_per_sequence, environment):

        min_action=self.robot.min_action*1000 #to be able to use randint
        max_action=self.robot.max_action*1000

        seqs=[]
        rewards=[]
        seqs_actions=[]

        for nb_seq in range(num_action_sequences):

            self.robot.state = np.array([0.5, 0.5])

            seq_positions=[self.robot.state]
            seq_actions=[]

            for nb_act in range(num_actions_per_sequence):
                act_1=np.random.randint(min_action,max_action)/1000
                act_2 = np.random.randint(min_action, max_action) / 1000
                act=np.array([act_1,act_2])

                self.robot.take_action(act,environment)

                seq_positions.append(self.robot.state)
                seq_actions.append(act)


            final_position=self.robot.state
            goal_position=environment.goal_state

            if goal_position.tolist()==final_position.tolist():
                reward=np.inf
                break
            else:
                distance=np.sqrt((final_position[0]-goal_position[0])**2+(final_position[1]-goal_position[1])**2)
                reward=1/distance

            rewards.append(reward)

            seqs.append(seq_positions)
            seqs_actions.append(seq_actions)




        seqs=np.array(seqs)

        #print(seqs)

        best_seq=seqs[np.argmax(rewards)]

        self.policy=seqs_actions[np.argmax(rewards)]

        return seqs,best_seq


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
        [self.seqs,self.best_seq]=self.agent.calculate_policy_with_random_shooting(num_action_sequences=1000, num_actions_per_sequence=30,
                                                         environment=self.environment)


    # on_update is called once per loop and is used to update the robot / environment
    def on_update(self, delta_time=90):

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



        for seq in self.seqs:
            if seq.tolist()!=self.best_seq.tolist():
                positions=[]
                for position in seq:

                    pos=[settings.SCREEN_SIZE * position[0],settings.SCREEN_SIZE * position[1]]
                    positions.append(pos)


                arcade.draw_line_strip(point_list=positions,color=[180,180,180])


        positions=[]
        for position in self.best_seq:

            pos=[settings.SCREEN_SIZE * position[0],settings.SCREEN_SIZE * position[1]]
            positions.append(pos)
            arcade.draw_circle_filled(pos[0],pos[1],radius=5, color=[0,255,0])
        arcade.draw_line_strip(point_list=positions,color=[0,255,0],line_width=4)



        pass


# The main entry point
if __name__ == "__main__":

    # Create a new program, which will also do the robot's initial planning
    MainProgram()

    # Run the main Arcade loop forever
    # This will repeatedly call the MainProgram.on_update() and MainProgram.on_draw() functions.
    arcade.run()
