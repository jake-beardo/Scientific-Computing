# Scientific-Computing

**Initial value problems for ODE's**
Found the solution at t = 5 for \dot{x} = x where x(0) = 1 to be approximately 148 using the Euler and the 4th order Runge-Kutta(RK4) methods. Using the analytical solution x(t) = e^t I found this to be true. The larger the step size the smaller the approximation due to this function diverging at t tends to infinity. The graph below illustrates this

![First Order approximation](Graphs/Approx_first_order.png)
Format: ![Alt Text](url)

A double log error graph is also plotted below showing that for Euler as step size increases as does error due to truncation error. Whereas, for RK4 there is an initial rounding error for small step sizes which then decreases up to an optimal point. The error then begins to increase with step size due to truncation error similarly to the Euler method.

![Error Graph](Graphs/errorgraph.png)
Format: ![Alt Text](url)

I then altered my code so that it could handle second order ODE's by changing the use of the ODE parameter in my functions so that they could handle a list of two ODE's. The way this is done is by manually splitting my second order ODE into two first order ODEs. These can then be used simultaneously within my solve_to function that calculates the Euler or RK4 approximation at each step. The plot below shows the Euler and RK4 approximation of \ddot{x} = - x. Which is equivalent to \dot{x} = y and \dot{y} = -x.

WHAT HAPPENS WITH LARGE T

![Second Order approximation](Graphs/approx_second_order.png)


**ODE problems without initial conditions**
The Lokta-Volterra equation is a predator prey equation where the red line in the graph below shows the predator and the orange the prey. The equation shows oscillations in the size of the population overtime.


![Lokta-Volterra (b=0.26)](Graphs/approx_second_order.png)
Format: ![Alt Text](url)


When b < 0.26 the population will decline very slowly overtime due to either the mortality rate being low or birth rate being high.

Whereas, when b > 0.26 the population will decrease at a much quicker rate as mortality rate may be very high or birth rate may be very low.

NB: I tested various b values and found anywhere less than or equal to 0.26 resulted in birufication meaning a stable solution.

Also, when we plot a birification diagram/phase portrait we witness a spiral when b > 0.26 and a blob when b < 0.26. No matter what initial conditions we use we will also end up with either of these two circumstances. This is shown below:

![Phase Portrait for varying values of b](Graphs/phase_port.png)
Format: ![Alt Text](url)

We can also plot the last values for predator and prey against different values of b as shown below. This plot shows us that b < 0.26 is unstable and beyond that the population becomes stable.
