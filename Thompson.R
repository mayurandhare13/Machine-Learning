# Reinforcement Learning
# Thompson Sampling

dataset <- read.csv("data/Ads_CTR_Optimisation.csv")

# Random Selection
N <- 10000
d <- 10
ads_selected = integer(0)
total_reward = 0
for (n in 1:N)
{
  ad = sample(1:10, 1)
  ads_selected = append(ads_selected, ad)
  reward = dataset[n, ad]
  total_reward = total_reward + reward
}

# Thompson Sampling

N <- 10000
d <- 10
ads_selected <- integer(0)
number_of_reward_1 <- integer(d)
number_of_reward_0 <- integer(d)
total_reward = 0

for (n in 1:N)
{
  ad = 0
  max_random = 0
  for (i in 1:d)
  {
    random_beta <- rbeta(n = 1, shape1 = number_of_reward_1[i] + 1,
                         shape2 = number_of_reward_0[i] + 1)
    if(random_beta > max_random)
    {
      max_random = random_beta
      ad = i
    }
  }
  
  ads_selected <- append(ads_selected, ad)
  reward <- dataset[n, ad]
  total_reward = total_reward + reward
  
  if(reward == 1)
  {
    number_of_reward_1[ad] = number_of_reward_1[ad] + 1
  }
  else
  {
    number_of_reward_0[ad] = number_of_reward_0[ad] + 1
  }
}

# visualization
hist(ads_selected, col = 'blue', main = 'Histogram of Ads Selected [Thompson Sampling]',
     xlab = 'Ads', ylab = 'Number of times ads selected')
