##Copyright (c) 2021 Peyman Sadrimajd##


import numpy as np
import scipy.optimize
import scipy.integrate
import copy
## unit for each parameter is commented after it is declared (inline)
## if the suggested value for the parameter is different -
## in the original ADM1 report by Batstone et al (2002) the original value is commented after the unit (inline)

##constant definition from the Rosen et al (2006) BSM2 report
R =  0.083145 #bar.M^-1.K^-1
T_base =  298.15 #K
P_atm =  1.013 #bar
T_op =  308.15 #k ##T_ad #=35 C

##parameter definition from the Rosen et al (2006) BSM2 report bmadm1_report
# Stoichiometric parameter
f_sI_xc =  0.1
f_xI_xc =  0.2
f_ch_xc =  0.2
f_pr_xc =  0.2
f_li_xc =  0.3
N_xc =  0.0376 / 14
N_I =  0.06 / 14 #kmole N.kg^-1COD
N_aa =  0.007 #kmole N.kg^-1COD
C_xc =  0.02786 #kmole C.kg^-1COD
C_sI =  0.03 #kmole C.kg^-1COD
C_ch =  0.0313 #kmole C.kg^-1COD
C_pr =  0.03 #kmole C.kg^-1COD
C_li =  0.022 #kmole C.kg^-1COD
C_xI =  0.03 #kmole C.kg^-1COD
C_su =  0.0313 #kmole C.kg^-1COD
C_aa =  0.03 #kmole C.kg^-1COD
f_fa_li =  0.95
C_fa =  0.0217 #kmole C.kg^-1COD
f_h2_su =  0.19
f_bu_su =  0.13
f_pro_su =  0.27
f_ac_su =  0.41
N_bac =  0.08 / 14 #kmole N.kg^-1COD
C_bu =  0.025 #kmole C.kg^-1COD
C_pro =  0.0268 #kmole C.kg^-1COD
C_ac =  0.0313 #kmole C.kg^-1COD
C_bac =  0.0313 #kmole C.kg^-1COD
Y_su =  0.1
f_h2_aa =  0.06
f_va_aa =  0.23
f_bu_aa =  0.26
f_pro_aa =  0.05
f_ac_aa =  0.40
C_va =  0.024 #kmole C.kg^-1COD
Y_aa =  0.08
Y_fa =  0.06
Y_c4 =  0.06
Y_pro =  0.04
C_ch4 =  0.0156 #kmole C.kg^-1COD
Y_ac =  0.05
Y_h2 =  0.06
##C_h2 and C_IN = 0 in equation 10 (S_IC)

# Biochemical parameter values from the Rosen et al (2006) BSM2 report
k_dis =  0.5 #d^-1
k_hyd_ch =  10 #d^-1
k_hyd_pr =  10 #d^-1
k_hyd_li =  10 #d^-1
K_S_IN =  10 ** -4 #M
k_m_su =  30 #d^-1
K_S_su =  0.5 #kgCOD.m^-3
pH_UL_aa =  5.5
pH_LL_aa =  4
k_m_aa =  50 #d^-1
K_S_aa =  0.3 ##kgCOD.m^-3
k_m_fa =  6 #d^-1
K_S_fa =  0.4 #kgCOD.m^-3
K_I_h2_fa =  5 * 10 ** -6 #kgCOD.m^-3
k_m_c4 =  20 #d^-1
K_S_c4 =  0.2 #kgCOD.m^-3
K_I_h2_c4 =  10 ** -5 #kgCOD.m^-3
k_m_pro =  13 #d^-1
K_S_pro =  0.1 #kgCOD.m^-3
K_I_h2_pro =  3.5 * 10 ** -6 #kgCOD.m^-3
k_m_ac =  8 #kgCOD.m^-3
K_S_ac =  0.15 #kgCOD.m^-3
K_I_nh3 =  0.0018 #M
pH_UL_ac =  7
pH_LL_ac =  6
k_m_h2 =  35 #d^-1
K_S_h2 =  7 * 10 ** -6 #kgCOD.m^-3
pH_UL_h2 =  6
pH_LL_h2 =  5
k_dec_X_su =  0.02 #d^-1
k_dec_X_aa =  0.02 #d^-1
k_dec_X_fa =  0.02 #d^-1
k_dec_X_c4 =  0.02 #d^-1
k_dec_X_pro =  0.02 #d^-1
k_dec_X_ac =  0.02 #d^-1
k_dec_X_h2 =  0.02 #d^-1
## M is kmole m^-3

# Physico-chemical parameter values from the Rosen et al (2006) BSM2 report
T_ad =  308.15 #K

K_w =  10 ** -14.0 * np.exp((55900 / (100 * R)) * (1 / T_base - 1 / T_ad)) #M #2.08 * 10 ^ -14

K_a_va =  10 ** -4.86 #M  ADM1 value = 1.38 * 10 ^ -5
K_a_bu =  10 ** -4.82 #M #1.5 * 10 ^ -5
K_a_pro =  10 ** -4.88 #M #1.32 * 10 ^ -5
K_a_ac =  10 ** -4.76 #M #1.74 * 10 ^ -5

K_a_co2 =  10 ** -6.35 * np.exp((7646 / (100 * R)) * (1 / T_base - 1 / T_ad)) #M #4.94 * 10 ^ -7
K_a_IN =  10 ** -9.25 * np.exp((51965 / (100 * R)) * (1 / T_base - 1 / T_ad)) #M #1.11 * 10 ^ -9

k_A_B_va =  10 ** 10 #M^-1 * d^-1
k_A_B_bu =  10 ** 10 #M^-1 * d^-1
k_A_B_pro =  10 ** 10 #M^-1 * d^-1
k_A_B_ac =  10 ** 10 #M^-1 * d^-1
k_A_B_co2 =  10 ** 10 #M^-1 * d^-1
k_A_B_IN =  10 ** 10 #M^-1 * d^-1

p_gas_h2o =  0.0313 * np.exp(5290 * (1 / T_base - 1 / T_ad)) #bar #0.0557
k_p = 5 * 10 ** 4 #m^3.d^-1.bar^-1 #only for BSM2 AD conditions, recalibrate for other AD cases #gas outlet friction
k_L_a =  200.0 #d^-1
K_H_co2 =  0.035 * np.exp((-19410 / (100 * R))* (1 / T_base - 1 / T_ad)) #Mliq.bar^-1 #0.0271
K_H_ch4 =  0.0014 * np.exp((-14240 / (100 * R)) * (1 / T_base - 1 / T_ad)) #Mliq.bar^-1 #0.00116
K_H_h2 =  7.8 * 10 ** -4 * np.exp(-4180 / (100 * R) * (1 / T_base - 1 / T_ad)) #Mliq.bar^-1 #7.38*10^-4

# Physical parameter values used in BSM2 from the Rosen et al (2006) BSM2 report
V_liq =  3400 #m^3
V_gas =  300 #m^3
V_ad = V_liq + V_gas #m^-3

##variable definition
# Steady-state input values (influent/feed) for BSM2 ADM1 from the Rosen et al (2006) BSM2 report
S_su_in = 0.01 #kg COD.m^-3
S_aa_in = 0.001 #kg COD.m^-3
S_fa_in = 0.001 #kg COD.m^-3
S_va_in = 0.001 #kg COD.m^-3
S_bu_in = 0.001 #kg COD.m^-3
S_pro_in = 0.001 #kg COD.m^-3
S_ac_in = 0.001 #kg COD.m^-3
S_h2_in = 10 ** -8  #kg COD.m^-3
S_ch4_in = 10 ** -5  #kg COD.m^-3
S_IC_in = 0.04 #kmole C.m^-3
S_IN_in = 0.01 #kmole N.m^-3
S_I_in = 0.02 #kg COD.m^-3

X_xc_in = 2.0 #kg COD.m^-3
X_ch_in = 5.0 #kg COD.m^-3
X_pr_in = 20.0 #kg COD.m^-3
X_li_in = 5.0 #kg COD.m^-3
X_su_in = 0.0 #kg COD.m^-3
X_aa_in = 0.01 #kg COD.m^-3
X_fa_in = 0.01 #kg COD.m^-3
X_c4_in = 0.01 #kg COD.m^-3
X_pro_in = 0.01 #kg COD.m^-3
X_ac_in = 0.01 #kg COD.m^-3
X_h2_in = 0.01 #kg COD.m^-3
X_I_in = 25.0 #kg COD.m^-3

S_cation_in = 0.04 #kmole.m^-3
S_anion_in = 0.02 #kmole.m^-3

Q_ad =  170.0 #m^-3.d^-1 flow rate

# SciPy ADM1 input array from Pettigrew (2017) jADM1 and Rosen et al (2006) BSM2 report
# initiate variables (initial values for the reactor state at t0)

S_su = 0.012 #kg COD.m^-3 monosaccharides
S_aa = 0.0053 #kg COD.m^-3 amino acids
S_fa = 0.099 #kg COD.m^-3 total long chain fatty acids
S_va = 0.012 #kg COD.m^-3 total valerate
S_bu = 0.013 #kg COD.m^-3 total butyrate
S_pro = 0.016 #kg COD.m^-3 total propionate
S_ac = 0.2 #kg COD.m^-3 total acetate
S_h2 = 2.30 * 10 ** -7 #kg COD.m^-3 hydrogen gas
S_ch4 = 0.055 #kg COD.m^-3 methane gas
S_IC = 0.15 #kmole C.m^-3 inorganic carbon
S_IN = 0.13 #kmole N.m^-3 inorganic nitrogen
S_I = 0.33 #kg COD.m^-3 soluble inerts

X_xc = 0.31 #kg COD.m^-3 composites
X_ch = 0.028 #kg COD.m^-3 carbohydrates
X_pr = 0.1 #kg COD.m^-3 proteins
X_li = 0.029 #kg COD.m^-3 lipids
X_su = 0.42 #kg COD.m^-3 sugar degraders
X_aa = 1.18 #kg COD.m^-3 amino acid degraders
X_fa = 0.24 #kg COD.m^-3 LCFA degraders
X_c4 = 0.43 #kg COD.m^-3 valerate and butyrate degraders
X_pro = 0.14 #kg COD.m^-3 propionate degraders
X_ac = 0.76 #kg COD.m^-3 acetate degraders
X_h2 = 0.32 #kg COD.m^-3 hydrogen degraders
X_I = 25.6 #kg COD.m^-3 particulate inerts

S_cation = 0.040 #kmole.m^-3 cations (metallic ions, strong base)
S_anion = 0.020 #kmole.m^-3 anions (metallic ions, strong acid)


pH = 7.4655377
S_H_ion = 0.00000003423 #kmole H.m^-3
S_va_ion = 0.011 #kg COD.m^-3 valerate
S_bu_ion = 0.013 #kg COD.m^-3 butyrate
S_pro_ion = 0.016 #kg COD.m^-3 propionate
S_ac_ion = 0.2 #kg COD.m^-3 acetate
S_hco3_ion = 0.14 #kmole C.m^-3 bicarbonate
S_nh3 = 0.0041 #kmole N.m^-3 ammonia
#S_nh4_ion = 0.126138 #kmole N.m^-3 the initial value is from Rosen et al (2006) BSM2 report and it is calculated further down and does not need to be initiated
#S_co2 = 0.0093003 #kmole C.m^-3 the initial value is from Rosen et al (2006) BSM2 report and it is calculated further down and does not need to be initiated
S_gas_h2 = 1.02 * 10 ** -5 #kg COD.m^-3 hydrogen concentration in gas phase
S_gas_ch4 = 1.63 #kg COD.m^-3 methane concentration in gas phase
S_gas_co2 = 0.014 #kmole C.m^-3 carbon dioxide concentration in gas phas


# related to pH inhibition taken from BSM2 report, they are global variables to avoid repeating them in DAE part
K_pH_aa =  (10 ** (-1 * (pH_LL_aa + pH_UL_aa) / 2.0))
nn_aa =  (3.0 / (pH_UL_aa - pH_LL_aa)) #we need a differece between N_aa and n_aa to avoid typos and nn_aa refers to n_aa in BSM2 report
K_pH_ac =  (10 ** (-1 * (pH_LL_ac + pH_UL_ac) / 2.0))
n_ac =  (3.0 / (pH_UL_ac - pH_LL_ac))
K_pH_h2 =  (10 ** (-1 * (pH_LL_h2 + pH_UL_h2) / 2.0))
n_h2 =  (3.0 / (pH_UL_h2 - pH_LL_h2))

S_nh4_ion =  0

S_co2 =  0

#pH equation
pH = - np.log10(S_H_ion)

States_AD_zero = [S_su,
             S_aa,
             S_fa,
             S_va,
             S_bu,
             S_pro,
             S_ac,
             S_h2,
             S_ch4,
             S_IC,
             S_IN,
             S_I,
             X_xc,
             X_ch,
             X_pr,
             X_li,
             X_su,
             X_aa,
             X_fa,
             X_c4,
             X_pro,
             X_ac,
             X_h2,
             X_I,
             S_cation,
             S_anion,
             S_H_ion,
             S_va_ion,
             S_bu_ion,
             S_pro_ion,
             S_ac_ion,
             S_hco3_ion,
             S_co2,
             S_nh3,
             S_nh4_ion,
             S_gas_h2,
             S_gas_ch4,
             S_gas_co2]

States_AD_input = [S_su_in,
             S_aa_in,
             S_fa_in,
             S_va_in,
             S_bu_in,
             S_pro_in,
             S_ac_in,
             S_h2_in,
             S_ch4_in,
             S_IC_in,
             S_IN_in,
             S_I_in,
             X_xc_in,
             X_ch_in,
             X_pr_in,
             X_li_in,
             X_su_in,
             X_aa_in,
             X_fa_in,
             X_c4_in,
             X_pro_in,
             X_ac_in,
             X_h2_in,
             X_I_in,
             S_cation_in,
             S_anion_in]

state_input = copy.deepcopy(States_AD_input)
state_zero = copy.deepcopy(States_AD_zero)
