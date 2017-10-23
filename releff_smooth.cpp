#include <TMB.hpp>
#include <iostream>

#define see(object) std::cout << #object ":\n" << object << "\n";

template<class Type>
Type mymvnorm(vector<Type> x, vector<Type> mu, vector<Type> sd, vector<Type> cor_par, int do_log)
{
  int dim = x.size();
  matrix<Type> L(dim,dim), Sigma(dim,dim);
  L.fill(0.0);
	int l = 0;
	for(int j = 0; j < dim; j++){
		for(int k = 0; k <= j; k++)
		{
			if(k == j) L(j,k) = 1.0;
		 	else
		 	{ 
				L(j,k) = cor_par(l);
				l += 1;
			}
	 	}
	 	Type normj = 0.0; L(j,0)*L(j,0);
	 	for(int k = 0; k <= j; k++) normj += L(j,k)*L(j,k);
	 	normj = exp(0.5*log(normj)); 
	 	for(int k = 0; k <= j; k++) L(j,k) = L(j,k)/normj;
		//L(j)(1,j) /= norm(L(j)(1,j));
 	}
 	L = L * L.transpose(); //correlation matrix
	for(int j = 0; j < dim; j++){
		for(int k = 0; k < dim; k++){
			Sigma(j,k) = L(j,k)*sd(j)*sd(k);
		}
	}
	Type cond_mu;
	Type cond_var;
	Type d = dnorm(x(0), mu(0), sd(0),1);
	for(int j = 1; j < dim; j++){
		cond_mu = 0.0 + (x(j-1) - mu(j-1))*Sigma(j,j-1)/Sigma(j-1,j-1);
		cond_var = Sigma(j,j) - Sigma(j,j-1)*Sigma(j,j-1)/Sigma(j-1,j-1);
		d += dnorm(x(j), cond_mu, exp(0.5*log(cond_var)),1);
	}
	if(do_log == 1) return(d);
	else return(exp(d));
}
   
template<class Type>
Type dbetabinom(Type x, Type n, Type mu, Type phi, int do_log)
{
  Type ll = lgamma(n + 1.0) - lgamma(x + 1.0) - lgamma(n - x + 1.0) + 
	  lgamma(x + mu*phi) + lgamma(n - x +(1-mu)*phi) - lgamma(n + phi) +
	  lgamma(phi) - lgamma(mu*phi) - lgamma((1-mu)*phi);
  if(do_log == 1) return(ll);
  else return(exp(ll));  
}

template<class Type>
Type objective_function<Type>::operator() ()
{

  //init_int n_sta
  DATA_IVECTOR(n_per_sta); //n_sta
  //int nobs
  DATA_VECTOR(n); //nobs 
  DATA_VECTOR(big); //nobs
  DATA_VECTOR(offst); //nobs

  DATA_MATRIX(XF); //nobs x n_beta_fixed
  DATA_MATRIX(XR); //nobs x n_smooth_reff
  DATA_MATRIX(ZF); //nobs x n_beta_reff
  DATA_MATRIX(ZR); //nobs x n_smooth_reff
  DATA_VECTOR(Dplus_diag); // n_smooth_reff
  DATA_VECTOR(rt_Dplus_diag); // n_smooth_reff

  DATA_MATRIX(phi_XF); //nobs x n_phi_betas
  DATA_MATRIX(phi_XR); //nobs x n_phi_smooth_reff
  DATA_MATRIX(phi_ZF); //nobs x n_phi_betas_reff
  DATA_MATRIX(phi_ZR); //nobs x n_phi_smooth_reff
  DATA_VECTOR(phi_Dplus_diag); //n_phi_smooth_reff
  DATA_VECTOR(phi_rt_Dplus_diag); //n_phi_smooth_reff

  DATA_INTEGER(use_Z);
  DATA_INTEGER(use_binomial);  //1 = yes, 0 = use beta-binomial
  DATA_INTEGER(use_beta_reff);
  DATA_INTEGER(use_mean_smooth_reff);
  DATA_INTEGER(use_smooth_reff);
  DATA_INTEGER(use_lambda_reff);
  DATA_INTEGER(use_phi_mean_smooth_reff);
  DATA_INTEGER(n_pred);
 
  PARAMETER_VECTOR(betas); // n_beta_fixed //fixed effects
  PARAMETER_VECTOR(beta_reff_var_pars); //n_beta_reff,beta_reff_var_pars_phase) //variance of random effects for fixed effects part of station-specific smooth use chol decomp to ensure pos-def
  PARAMETER_VECTOR(beta_reff_cor_pars); //n_beta_reff_cor_pars,beta_reff_cor_pars_phase) //variance of random effects for fixed effects part of station-specific smooth use chol decomp to ensure pos-def
  PARAMETER(lambda_par); //log(mean) smoothing parameter across stations
  PARAMETER(lambda_par_stations); 
  PARAMETER(lambda_var_par); //log(variance) of smoothing parameter across stations
  PARAMETER_VECTOR(phi_betas); //n_phi_betas //fixed effects for log of beta-binomial dispersion parameter
  PARAMETER(phi_lambda_par); //fixed effects for log of beta-binomial dispersion parameter
  PARAMETER_MATRIX(beta_reff); //n_sta x n_beta_reff //random effects for fixed effects part of station-specific smooth
  PARAMETER_VECTOR(mean_smooth_reff); //n_smooth_reff // population-level average random effects portion of smoother
  PARAMETER_MATRIX(smooth_reff); //n_sta x n_smooth_reff //random effects part of station-specific smooth
  PARAMETER_VECTOR(lambda_reff); //n_sta //station-specific lambdas (random effects)
  PARAMETER_VECTOR(phi_mean_smooth_reff); //n_phi_smooth_reff // population-level average random effects portion of smoother
  Type nll = 0.0;
  vector<Type> eta(n.size()), phi_eta(n.size()), mu(n.size()), phi(n.size());
  using namespace density;
  
  //matrix<Type> beta_reff_cov_mat(n_beta_reff,n_beta_reff);
	//distribution of population-level random effects for smoother portion
	if(use_mean_smooth_reff == 1) {
	  for(int j = 0; j < mean_smooth_reff.size(); j++) nll -= dnorm(mean_smooth_reff(j), Type(0.0), exp(-0.5*(log(Dplus_diag(j)) + lambda_par)), 1);
	}
	//distribution of population-level random effects for smoother portion of phi (dispersion parameter)
	if(use_binomial == 0) if(use_phi_mean_smooth_reff == 1)
	{
	  for(int j = 0; j < phi_mean_smooth_reff.size(); j++) nll -= dnorm(phi_mean_smooth_reff(j), Type(0.0), exp(-0.5*(log(phi_Dplus_diag(j)) + phi_lambda_par)), 1);
	}
	//distribution of random effects on fixed effects portion of mean (intercept and some coefficients of smoother)
	//series of conditionals.
	if(use_beta_reff == 1) 
  {
    vector<Type> beta_reff_sds = exp(beta_reff_var_pars);
    /*if(beta_reff.cols()>1)
    {
      matrix<Type> beta_reff_cov_mat = UNSTRUCTURED_CORR(beta_reff_cor_pars).cov();
      //see(beta_reff_cov_mat);
      UNSTRUCTURED_CORR_t<Type> beta_reff_density(beta_reff_cor_pars);// mvnorm with variance = 1
      for(int i = 0; i <n_per_sta.size(); i++) 
      {
        vector<Type> beta_reff_i = beta_reff.row(i); 
        nll -= log(VECSCALE(beta_reff_density,beta_reff_sds)(beta_reff_i)); //mvnorm with correct variances
      }
  	  REPORT(beta_reff_cov_mat);
  	}
  	else
  	{
  	  for(int i = 1; i < beta_reff.cols(); i++) see(i);
      for(int i = 0; i <n_per_sta.size(); i++) nll -= dnorm(beta_reff(i,0), Type(0.0), beta_reff_sds(0),1);
    }*/
    for(int i = 0; i <n_per_sta.size(); i++) 
    {
      vector<Type> beta_reff_i = beta_reff.row(i);
      vector<Type> mu(beta_reff_i.size());
      mu.fill(0.0);
      nll -= mymvnorm(beta_reff_i,mu, beta_reff_sds,beta_reff_cor_pars,1);
    }
  }
  vector<Type> mean_smooth_reff_use = mean_smooth_reff;
  vector<Type> phi_mean_smooth_reff_use = phi_mean_smooth_reff;
  matrix<Type> XR_use = XR, phi_XR_use = phi_XR;  
  if(use_Z == 1)
  {
    XR_use = ZR;
    mean_smooth_reff_use  = rt_Dplus_diag * mean_smooth_reff;
    if(use_binomial == 0) 
    {
      phi_XR_use = phi_ZR;
      phi_mean_smooth_reff_use  = rt_Dplus_diag * phi_mean_smooth_reff;
    }
  }
   
  int ii = 0;
  for(int i = 0; i < n_per_sta.size(); i++) 
  {
	  //distribution of random effects portion of smoother given lambda at the station
    if(use_smooth_reff == 1)
    {
      vector<Type> sd_smooth_reff_i = exp(Type(-0.5) * (log(Dplus_diag) +  lambda_reff(i) + lambda_par_stations + lambda_par)); 
      for(int j = 0; j < smooth_reff.cols(); j++) nll -= dnorm(smooth_reff(i,j), Type(0.0), sd_smooth_reff_i(j),1);
      
    }
    if(use_lambda_reff == 1) nll -= dnorm(lambda_reff(i), Type(0.0), exp(lambda_var_par),1);
	  vector<Type> smooth_reff_use = smooth_reff.row(i);
   	if(use_Z == 1) smooth_reff_use = rt_Dplus_diag * smooth_reff_use;
		vector<Type> beta_reff_i = beta_reff.row(i);
   	
		for(int j = 0; j < n_per_sta(i); j++) 
		{
		  vector<Type> XFj = XF.row(ii+j);
		  vector<Type> ZFj = ZF.row(ii+j);
		  vector<Type> XRj = XR_use.row(ii+j);
		  eta(ii + j) = offst(ii + j) + (XFj * betas).sum() + (ZFj * beta_reff_i).sum() + (XRj * (mean_smooth_reff_use + smooth_reff_use)).sum();
		  mu(ii + j) = 1.0/(1.0 + exp(-eta(ii+j)));
		  if(use_binomial == 1) nll -= dbinom(big(ii+j),n(ii+j),mu(ii+j),1);
		  else 
		  {
		    vector<Type> phi_XFj = phi_XF.row(ii+j);
		    vector<Type> phi_XRj = phi_XR_use.row(ii+j);
			  phi_eta(ii+j) = (phi_XFj * phi_betas).sum() + (phi_XRj * phi_mean_smooth_reff_use).sum();
  		  phi(ii+j) = exp(phi_eta(ii+j));
		    nll -= dbetabinom(big(ii+j),n(ii+j),mu(ii+j),phi(ii+j),1);
		  }
		}
    ii += n_per_sta(i);
  }
	REPORT(eta);
	REPORT(mu);
	if(use_binomial == 0)
	{
	  REPORT(phi_eta);
	  REPORT(phi);
	}
	if(n_pred>0)
	{
	  int k = n_per_sta.sum();
    vector<Type> mean_pred_eta(n_pred);	  
	  matrix<Type> station_pred_eta(n_per_sta.size(),n_pred);
	  for(int j = 0; j < n_pred; j++)
	  {
		  vector<Type> XFj = XF.row(k+j);
		  vector<Type> ZFj = ZF.row(k+j);
		  vector<Type> XRj = XR_use.row(k+j);
      mean_pred_eta(j) = (XFj * betas).sum() + (XRj * mean_smooth_reff_use).sum();
      for(int i = 0; i < n_per_sta.size(); i++) 
      {
	      vector<Type> smooth_reff_pred = smooth_reff.row(i);
		    vector<Type> beta_reff_pred = beta_reff.row(i);
		    station_pred_eta(i,j) = mean_pred_eta(j) + (ZFj * beta_reff_pred).sum() + (XRj * smooth_reff_pred).sum();
      }
    }
    REPORT(station_pred_eta);
    ADREPORT(mean_pred_eta);
    if(use_binomial == 0)
    {
  	  vector<Type> mean_pred_phi_eta(n_pred);
	    for(int j = 0; j < n_pred; j++)
	    {
		    vector<Type> phi_XFj = phi_XF.row(k+j);
		    vector<Type> phi_XRj = phi_XR_use.row(k+j);
  	    mean_pred_phi_eta(j) = (phi_XFj * phi_betas).sum() + (phi_XRj * phi_mean_smooth_reff_use).sum();
      }
      ADREPORT(mean_pred_phi_eta);
    }
  }
	return(nll);
}

