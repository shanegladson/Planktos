#! /usr/bin/env python3

import sys

sys.path.append('../..')
import planktos
import numpy as np

# As before, we begin by creating a default environment. It does not matter what 
#   the dimensions are - when we load the vtk data, it will automatically 
#   change shape to adjust to the data!
envir = planktos.environment()

# Cylinder Geometry Data (ALWAYS UPDATE WITH IB2D RUN DATA)
xCenter = 0.25  # (m)
yCenter = 0.1  # (m)
radius = 0.02  # (m)

# Swarm parameters
detectionDistance = 3.35 * (
        2 * radius) + radius  # Distance at which wasps can detect the target (add the radius since this is the center of the circle)
burstSpeed = 0.34  # Maximum flight speed of the wasps (m/s)
enduranceTime = 0.6  # Time (s) of endurance for burst speed
cruiseSpeedFactor = 0.7  # Multiplier of burst speed at which the wasps normally fly
exhaustedSpeedFactor = 0.1  # Multiplier of burst speed when wasps are exhausted from locked-in state
referenceVector = (1, 0)  # Reference vector when calculating angle relative to agent, must be in direction of flow
adjustingForAngle = False  # Set to True if you want the agents to compensate for fluid flow
includeJitter = True  # Set to True if you want to include Brownian motion in the simulation
directionalNoise = True  # Set to True if you want to include imperfect directional locking onto the target
directionalStdDev = np.pi / 6  # Standard deviation in the normal distribution when using directionalNoise
updateDirectionRate = 2  # The number of time steps between changes in direction for a wasp
fluidRecognitionLag = updateDirectionRate  # The delay in time steps for the wasp to recognize the fluid drift (this means compensation for fluid flow is delayed by a few steps)

# Starting swarm  size, dt, and run time
swarm_size = 100
dt = 0.005
runsteps = 100
runtime = runsteps * dt

# Read in ib2d vtk data (file, dt, print_dump)
envir.read_IB2d_vtk_data('ib2d_data', 1.0e-5, 400)

# Now we read in the vertex data to get an immersed mesh
envir.read_IB2d_vertex_data('ib2d_data/viv_geo.vertex')


class WaspSwarm(planktos.swarm):

    # This get_positions method overrides the planktos.swarm get_positions method
    def get_positions(self, dt, params=None):

        # Get fluid drift vectors at every agent location
        fluiddrift = self.get_fluid_drift()

        # Get stuck property
        stick = self.get_prop('stick')

        # Get timeoflock property
        timelock = self.get_prop('timeoflock')

        # Get lockedin property
        lockedin = self.get_prop('lockedin')

        # Check if it is time to update the flight direction (divide time by dt to get the number of time steps)
        if np.around(self.envir.time / dt, decimals=0) % updateDirectionRate == 0:
            # Create an array of target coordinates at the center of the cylinder
            targets = np.array([(xCenter, yCenter) for i in range(swarm_size)])

            # Find the vector from the agent positions to the target
            vectors = targets - self.positions

            # Get locked in condition by finding agents within the detection distance (assume false first, then add trues to array)
            lockedin += (np.linalg.norm(vectors, axis=1) < detectionDistance)

            # Subtract the fluid drift vector if adjusting for fluid flow, otherwise leave the vector as is
            # TODO: Fix the adjustingForAngle to correctly adjust when agents are upwind of the target (leave False for now)
            if adjustingForAngle:
                pastFluidDrift = self.get_fluid_drift(time=(self.envir.time - fluidRecognitionLag * dt))
                vectors -= pastFluidDrift

            # Normalize the vectors
            denom = np.tile(np.linalg.norm(vectors, axis=1),
                            (len(self.envir.L), 1)).T
            normvec = vectors / denom

            # Add directional noise by selecting an angle from a normal distribution and creating a new movement vector
            if directionalNoise:
                # Arctan (normvec y, normvec x)
                currentAngle = np.arctan2(normvec[:, 1], normvec[:, 0])
                newAngles = np.random.normal(loc=currentAngle, scale=directionalStdDev)
                normvec = np.vstack((np.cos(newAngles), np.sin(newAngles))).T

                # Store the new directions for future use
                self.props['direction'] = newAngles

            else:
                # Arctan (normvec y, normvec x)
                currentAngle = np.arctan2(normvec[:, 1], normvec[:, 0])
                self.props['direction'] = currentAngle

        # If it isn't time to update direction, get the previous flight direction and continue flying in that direction
        else:
            pastAngles = self.get_prop('direction')
            normvec = np.vstack((np.cos(pastAngles), np.sin(pastAngles))).T

        # Begin the locked-in timer for agents that have detected the target and will use burst speed
        for i in range(swarm_size):
            if timelock[i] == runtime and lockedin[i]:
                timelock[i] = self.envir.time

        # Update the timeoflock and lockedin attribute for the swarm (don't forget!)
        self.props['timeoflock'] = timelock
        self.props['lockedin'] = lockedin

        # Find which agents are "exhausted" after having used their burst speed for the endurance time
        exhausted = (self.envir.time - timelock) > enduranceTime

        # TODO: Move the randomWalkAngles to being updated with the lockedin wasps, may need to create another attribute that stores randomWalkAngles
        # Calculate a random walk vector when not locked in on the target
        randomWalkAngles = np.random.uniform(0, 2 * np.pi, swarm_size)
        xComponent = np.cos(randomWalkAngles)
        yComponent = np.sin(randomWalkAngles)
        randomWalkVector = np.vstack((xComponent, yComponent)).T

        # Calculate the movement vectors based on whether the wasps have detected the cylinder, also including the exhausted condition
        movevec = np.logical_and(np.expand_dims(lockedin, 1), np.expand_dims(exhausted, 1)) * exhaustedSpeedFactor * (
                normvec * burstSpeed) + \
                  np.logical_and(np.expand_dims(lockedin, 1), np.expand_dims(~exhausted, 1)) * (normvec * burstSpeed) + \
                  np.expand_dims(~lockedin, 1) * randomWalkVector * (burstSpeed * cruiseSpeedFactor)

        # Returns the new positions including Euler Brownian motion if desired
        if includeJitter:
            vecwfluid = movevec + fluiddrift

            # Brownian motion to include jitter
            random = planktos.motion.Euler_brownian_motion(self, dt, mu=vecwfluid)

            # Consider the agents that are stuck to an immersed boundary
            return np.expand_dims(~stick, 1) * random + \
                   np.expand_dims(stick, 1) * self.positions

        else:
            # Consider fluid advection (vector with fluid), multiply by dt since the time step is <1
            vecwfluid = (movevec + fluiddrift) * dt

            # Consider the agents that are stuck to an immersed boundary
            return np.expand_dims(~stick, 1) * (vecwfluid + self.positions) + \
                   np.expand_dims(stick, 1) * self.positions


# Initializes the agents at random points within a circle of desired radius and location
randomRadius = 0.04 * np.random.uniform(0, 1, size=swarm_size)  # (m)
randomAngle = np.random.uniform(0, 2 * np.pi, size=swarm_size)
circle = np.vstack((randomRadius * np.cos(randomAngle), randomRadius * np.sin(randomAngle))).T
circleCenter = (0.40, 0.1)
initRegion = circle + circleCenter

# Initialize the swarm and add desired attributes
swrm = WaspSwarm(envir=envir, swarm_size=swarm_size, init=initRegion)

swrm.props['stick'] = np.full(swarm_size, False)  # Determines whether the wasps have reached the trap
swrm.props['timeoflock'] = np.full(swarm_size,
                                   runtime)  # Stores time of lock onto the target, used when calculating exhaustion
swrm.props['direction'] = np.random.uniform(0, 2 * np.pi,
                                            swarm_size)  # Stores the travel direction for the wasps, updated at the updateDirectionRate
swrm.props['lockedin'] = np.full(swarm_size,
                                 False)  # A longer term storage for remembering which wasps are locked in on the target

# Remember that variance is standard deviation squared, so this would be a standard deviation of 0.01 m (1 cm)
swrm.shared_props['cov'] *= 0.0001

# Now let's move the swarm with time step dt
print('Moving swarm...')
for ii in range(runsteps):
    swrm.move(dt, ib_collisions='sticky')

    # Check for sticking condition
    swrm.props['stick'] = np.logical_or(swrm.props['stick'], swrm.ib_collision)

# TODO: Fix to account for agents that were stuck on the boundary walls
print("The total number of stuck wasps is: ", swrm.get_prop('stick').sum())

swrm.plot_all(movie_filename='WaspTestV51R02E15DirectionNoise.mp4', fps=20, fluid='quiver')
