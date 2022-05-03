
data {
  // timeseries length
  int<lower = 1> n_days;
  
  // priors 
  real<lower = 0, upper = 1> tpp_mu_prior_mu;
  real<lower = 0> tpp_mu_prior_scale;
  real<lower = 0> tpp_sigma_prior_mu;
  real<lower = 0> tpp_sigma_prior_scale;
  real<lower = 0> tpp_coef_prior_scale;
  real<lower = 0> df_prior_alpha;
  real<lower = 0> df_prior_beta;
  
  // election data for polling
  int polls_n_elections;
  int polls_election_days[polls_n_elections]; // days on which  elections occur
  real polls_election_tpp_alp[polls_n_elections]; // historical election results
  
  // polling data
  // total number of polls
  int<lower=1> polls_n;
  
  // number of firms
  int<lower=1> polls_firms_n;
  
  // pollster index
  int<lower=1,upper=polls_firms_n> polls_firm[polls_n];
  
  // polling data
  real<lower=0,upper=1> polls_tpp_alp[polls_n];
  real<lower=0> polls_standard_error[polls_n];
  int<lower=1> polls_day[polls_n];
  //int<lower=1, upper=2> poll_close_to_next[polls_n];
  
  // polling standard error inflator
  real<lower=1> polls_se_inflator;
  
  // election data for tpp2seats
  int<lower=1> tpp2seats_n_elections;
  real tpp2seats_alpha_prior_mu;
  real<lower = 0> tpp2seats_alpha_prior_scale;
  real tpp2seats_beta_prior_mu;
  real<lower = 0> tpp2seats_beta_prior_scale;
  int tpp2seats_election_major_seats[tpp2seats_n_elections];
  int tpp2seats_election_alp_seats[tpp2seats_n_elections];
  vector<lower=0, upper=1>[tpp2seats_n_elections] tpp2seats_election_tpp_alp;
  
  // data for crossbench model
  int<lower=1> crossbench_n_elections;
  int<lower=0> crossbench_retain_shots[crossbench_n_elections];
  int crossbench_retained [crossbench_n_elections];
  int<lower=0> crossbench_gain_shots[crossbench_n_elections];
  int crossbench_gained [crossbench_n_elections];
  
}

parameters {
  
  // tpp_margin estimate
  vector<lower=0, upper=1>[n_days] tpp_mu;

  // tpp_margin sigma
  real<lower=0> tpp_sigma;

  // degrees of freedom between days
  real<lower=1, upper=7> student_t_df;

  // house effect
  real<lower=-1, upper=1> house_effect[polls_firms_n];

  // close to next effect
  real<lower=-1, upper=1> polling_bias;
  
  // tpp to seat
  real alpha;
  real beta;

  // crossbench
  real<lower=0, upper=1> crossbench_p_retain;
  real<lower=0, upper=1> crossbench_p_gain;
  
}

transformed parameters {
  // linear predictor
  vector[tpp2seats_n_elections] tpp2seats_y_hat;
  tpp2seats_y_hat = alpha + tpp2seats_election_tpp_alp * beta;
}

model {
  // priors # make parameters for these
  tpp_mu[1] ~ normal(tpp_mu_prior_mu, tpp_mu_prior_scale); // starting state space
  tpp_sigma ~ normal(tpp_sigma_prior_mu, tpp_sigma_prior_scale);     // prior for innovation sd.
  house_effect ~ normal(0, tpp_coef_prior_scale); // ie a fairly loose prior for house effects (on scale of [0,1])
  polling_bias ~ normal(0, tpp_coef_prior_scale);

  student_t_df ~ gamma(df_prior_alpha, df_prior_beta);

  // state model
  tpp_mu[2:n_days] ~ student_t(student_t_df, tpp_mu[1:(n_days - 1)], tpp_sigma);

  // historical election results
  for(election_i in 1:polls_n_elections){
    polls_election_tpp_alp[election_i] ~ normal(tpp_mu[polls_election_days[election_i]], 0.0001); // we know tpp_mu very accurately on election day
  }

  //polls
  for(poll_i in 1:polls_n){
      polls_tpp_alp[poll_i] ~ normal(tpp_mu[polls_day[poll_i]] + house_effect[polls_firm[poll_i]] + polling_bias,  polls_standard_error[poll_i] * polls_se_inflator);

  }

  //  //tpp2seats
  alpha ~ normal(tpp2seats_alpha_prior_mu, tpp2seats_alpha_prior_scale);
  beta ~ normal(tpp2seats_beta_prior_mu, tpp2seats_beta_prior_scale);

  tpp2seats_election_alp_seats ~ binomial_logit(tpp2seats_election_major_seats, tpp2seats_y_hat);

  // crossbench
  crossbench_p_retain ~ beta(5,2);

  crossbench_retained ~ binomial(crossbench_retain_shots, crossbench_p_retain);

  crossbench_p_gain ~ beta(1,5);
  crossbench_gained ~ binomial(crossbench_gain_shots, crossbench_p_gain);
}

// generated quantities {
//   int<lower=0, upper=151> crossbench_retain;
//   int<lower=0, upper=151> crossbench_gain;
//   int<lower=0, upper=151> crossbench_total;
//   int<lower=0, upper=151> majors_seats;
//   real<lower=0, upper=151> alp_seats;
//   real<lower=0, upper=151> lnc_seats;
//   real<lower=-151, upper=151> seat_margin;
//   real<lower=0, upper=1> election_day_p_alp_win_seat;
//   real election_day_tpp_alp;
//   real election_day_tpp_margin;
// 
//   crossbench_retain = binomial_rng(7, crossbench_p_retain);
// 
//   crossbench_gain = binomial_rng(144, crossbench_p_gain);
// 
//   crossbench_total = crossbench_retain + crossbench_gain;
// 
//   majors_seats = 151 - crossbench_total;
// 
//   election_day_tpp_alp = tpp_mu[n_days];
// 
//   election_day_tpp_margin = (1 - election_day_tpp_alp) - election_day_tpp_alp;
// 
//   election_day_p_alp_win_seat = inv_logit(alpha + beta * election_day_tpp_alp);
// 
//   alp_seats = binomial_rng(majors_seats, election_day_p_alp_win_seat);
// 
//   lnc_seats = majors_seats - alp_seats;
// 
//   seat_margin = lnc_seats - alp_seats;
// 
// }
