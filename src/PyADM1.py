from Parameters import *

def ADM1_ODE(t, state_zero):

  S_su = state_zero[0]
  S_aa = state_zero[1]
  S_fa = state_zero[2]
  S_va = state_zero[3]
  S_bu = state_zero[4]
  S_pro = state_zero[5]
  S_ac = state_zero[6]
  S_h2 = state_zero[7]
  S_ch4 = state_zero[8]
  S_IC = state_zero[9]
  S_IN = state_zero[10]
  S_I = state_zero[11]
  X_xc = state_zero[12]
  X_ch = state_zero[13]
  X_pr = state_zero[14]
  X_li = state_zero[15]
  X_su = state_zero[16]
  X_aa = state_zero[17]
  X_fa = state_zero[18]
  X_c4 = state_zero[19]
  X_pro = state_zero[20]
  X_ac = state_zero[21]
  X_h2 = state_zero[22]
  X_I = state_zero[23]
  S_cation =  state_zero[24]
  S_anion = state_zero[25]
  S_H_ion =  state_zero[26]
  S_va_ion = state_zero[27]
  S_bu_ion = state_zero[28]
  S_pro_ion = state_zero[29]
  S_ac_ion = state_zero[30]
  S_hco3_ion =  state_zero[31]
  S_co2 = state_zero[32]
  S_nh3 = state_zero[33]
  S_nh4_ion =  state_zero[34]
  S_gas_h2 = state_zero[35]
  S_gas_ch4 = state_zero[36]
  S_gas_co2 = state_zero[37]


  S_su_in = state_input[0]
  S_aa_in = state_input[1]
  S_fa_in = state_input[2]
  S_va_in = state_input[3]
  S_bu_in =  state_input[4]
  S_pro_in =  state_input[5]
  S_ac_in =  state_input[6]
  S_h2_in =   state_input[7]
  S_ch4_in = state_input[8]
  S_IC_in = state_input[9]
  S_IN_in =  state_input[10]
  S_I_in = state_input[11]
  X_xc_in =  state_input[12]
  X_ch_in = state_input[13]
  X_pr_in = state_input[14]
  X_li_in =  state_input[15]
  X_su_in =  state_input[16]
  X_aa_in =  state_input[17]
  X_fa_in =  state_input[18]
  X_c4_in =  state_input[19]
  X_pro_in =  state_input[20]
  X_ac_in =  state_input[21]
  X_h2_in =  state_input[22]
  X_I_in = state_input[23]
  S_cation_in = state_input[24]
  S_anion_in = state_input[25]

  S_nh4_ion =  (S_IN - S_nh3)

  S_co2 =  (S_IC - S_hco3_ion)

  I_pH_aa =  ((K_pH_aa ** nn_aa) / (S_H_ion ** nn_aa + K_pH_aa ** nn_aa))
  I_pH_ac =  ((K_pH_ac ** n_ac) / (S_H_ion ** n_ac + K_pH_ac ** n_ac))
  I_pH_h2 =  ((K_pH_h2 ** n_h2) / (S_H_ion ** n_h2 + K_pH_h2 ** n_h2))
  I_IN_lim =  (1 / (1 + (K_S_IN / S_IN)))
  I_h2_fa =  (1 / (1 + (S_h2 / K_I_h2_fa)))
  I_h2_c4 =  (1 / (1 + (S_h2 / K_I_h2_c4)))
  I_h2_pro =  (1 / (1 + (S_h2 / K_I_h2_pro)))
  I_nh3 =  (1 / (1 + (S_nh3 / K_I_nh3)))

  I_5 =  (I_pH_aa * I_IN_lim)
  I_6 = I_5
  I_7 =  (I_pH_aa * I_IN_lim * I_h2_fa)
  I_8 =  (I_pH_aa * I_IN_lim * I_h2_c4)
  I_9 = I_8
  I_10 =  (I_pH_aa * I_IN_lim * I_h2_pro)
  I_11 =  (I_pH_ac * I_IN_lim * I_nh3)
  I_12 =  (I_pH_h2 * I_IN_lim)

  # biochemical process rates from Rosen et al (2006) BSM2 report
  Rho_1 =  (k_dis * X_xc)  # Disintegration
  Rho_2 =  (k_hyd_ch * X_ch)  # Hydrolysis of carbohydrates
  Rho_3 =  (k_hyd_pr * X_pr)  # Hydrolysis of proteins
  Rho_4 =  (k_hyd_li * X_li)  # Hydrolysis of lipids
  Rho_5 =  (k_m_su * (S_su / (K_S_su + S_su)) * X_su * I_5)  # Uptake of sugars
  Rho_6 =  (k_m_aa * (S_aa / (K_S_aa + S_aa)) * X_aa * I_6)  # Uptake of amino-acids
  Rho_7 =  (k_m_fa * (S_fa / (K_S_fa + S_fa)) * X_fa * I_7)  # Uptake of LCFA (long-chain fatty acids)
  Rho_8 =  (k_m_c4 * (S_va / (K_S_c4 + S_va )) * X_c4 * (S_va / (S_bu + S_va + 1e-6)) * I_8)  # Uptake of valerate
  Rho_9 =  (k_m_c4 * (S_bu / (K_S_c4 + S_bu )) * X_c4 * (S_bu / (S_bu + S_va + 1e-6)) * I_9)  # Uptake of butyrate
  Rho_10 =  (k_m_pro * (S_pro / (K_S_pro + S_pro)) * X_pro * I_10)  # Uptake of propionate
  Rho_11 =  (k_m_ac * (S_ac / (K_S_ac + S_ac)) * X_ac * I_11)  # Uptake of acetate
  Rho_12 =  (k_m_h2 * (S_h2 / (K_S_h2 + S_h2)) * X_h2 * I_12)  # Uptake of hydrogen
  Rho_13 =  (k_dec_X_su * X_su)  # Decay of X_su
  Rho_14 =  (k_dec_X_aa * X_aa)  # Decay of X_aa
  Rho_15 =  (k_dec_X_fa * X_fa)  # Decay of X_fa
  Rho_16 =  (k_dec_X_c4 * X_c4)  # Decay of X_c4
  Rho_17 =  (k_dec_X_pro * X_pro)  # Decay of X_pro
  Rho_18 =  (k_dec_X_ac * X_ac)  # Decay of X_ac
  Rho_19 =  (k_dec_X_h2 * X_h2)  # Decay of X_h2

  # acid-base rates for the BSM2 ODE implementation from Rosen et al (2006) BSM2 report
  Rho_A_4 =  (k_A_B_va * (S_va_ion * (K_a_va + S_H_ion) - K_a_va * S_va))
  Rho_A_5 =  (k_A_B_bu * (S_bu_ion * (K_a_bu + S_H_ion) - K_a_bu * S_bu))
  Rho_A_6 =  (k_A_B_pro * (S_pro_ion * (K_a_pro + S_H_ion) - K_a_pro * S_pro))
  Rho_A_7 =  (k_A_B_ac * (S_ac_ion * (K_a_ac + S_H_ion) - K_a_ac * S_ac))
  Rho_A_10 =  (k_A_B_co2 * (S_hco3_ion * (K_a_co2 + S_H_ion) - K_a_co2 * S_IC))
  Rho_A_11 =  (k_A_B_IN * (S_nh3 * (K_a_IN + S_H_ion) - K_a_IN * S_IN))

  # gas phase algebraic equations from Rosen et al (2006) BSM2 report
  p_gas_h2 =  (S_gas_h2 * R * T_op / 16)
  p_gas_ch4 =  (S_gas_ch4 * R * T_op / 64)
  p_gas_co2 =  (S_gas_co2 * R * T_op)


  P_gas =  (p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_gas_h2o)
  q_gas =  (k_p * (P_gas - P_atm))
  if q_gas < 0:    q_gas = 0


  # gas transfer rates from Rosen et al (2006) BSM2 report
  Rho_T_8 =  (k_L_a * (S_h2 - 16 * K_H_h2 * p_gas_h2))
  Rho_T_9 =  (k_L_a * (S_ch4 - 64 * K_H_ch4 * p_gas_ch4))
  Rho_T_10 =  (k_L_a * (S_co2 - K_H_co2 * p_gas_co2))

  ##differential equaitons from Rosen et al (2006) BSM2 report
  # differential equations 1 to 12 (soluble matter)
  diff_S_su = Q_ad / V_liq * (S_su_in - S_su) + Rho_2 + (1 - f_fa_li) * Rho_4 - Rho_5  # eq1

  diff_S_aa = Q_ad / V_liq * (S_aa_in - S_aa) + Rho_3 - Rho_6  # eq2

  diff_S_fa = Q_ad / V_liq * (S_fa_in - S_fa) + (f_fa_li * Rho_4) - Rho_7  # eq3

  diff_S_va = Q_ad / V_liq * (S_va_in - S_va) + (1 - Y_aa) * f_va_aa * Rho_6 - Rho_8  # eq4

  diff_S_bu = Q_ad / V_liq * (S_bu_in - S_bu) + (1 - Y_su) * f_bu_su * Rho_5 + (1 - Y_aa) * f_bu_aa * Rho_6 - Rho_9  # eq5

  diff_S_pro = Q_ad / V_liq * (S_pro_in - S_pro) + (1 - Y_su) * f_pro_su * Rho_5 + (1 - Y_aa) * f_pro_aa * Rho_6 + (1 - Y_c4) * 0.54 * Rho_8 - Rho_10  # eq6

  diff_S_ac = Q_ad / V_liq * (S_ac_in - S_ac) + (1 - Y_su) * f_ac_su * Rho_5 + (1 - Y_aa) * f_ac_aa * Rho_6 + (1 - Y_fa) * 0.7 * Rho_7 + (1 - Y_c4) * 0.31 * Rho_8 + (1 - Y_c4) * 0.8 * Rho_9 + (1 - Y_pro) * 0.57 * Rho_10 - Rho_11  # eq7

  #diff_S_h2 below with DAE paralel equaitons

  diff_S_ch4 = Q_ad / V_liq * (S_ch4_in - S_ch4) + (1 - Y_ac) * Rho_11 + (1 - Y_h2) * Rho_12 - Rho_T_9  # eq9


  ## eq10 ##
  s_1 =  (-1 * C_xc + f_sI_xc * C_sI + f_ch_xc * C_ch + f_pr_xc * C_pr + f_li_xc * C_li + f_xI_xc * C_xI)
  s_2 =  (-1 * C_ch + C_su)
  s_3 =  (-1 * C_pr + C_aa)
  s_4 =  (-1 * C_li + (1 - f_fa_li) * C_su + f_fa_li * C_fa)
  s_5 =  (-1 * C_su + (1 - Y_su) * (f_bu_su * C_bu + f_pro_su * C_pro + f_ac_su * C_ac) + Y_su * C_bac)
  s_6 =  (-1 * C_aa + (1 - Y_aa) * (f_va_aa * C_va + f_bu_aa * C_bu + f_pro_aa * C_pro + f_ac_aa * C_ac) + Y_aa * C_bac)
  s_7 =  (-1 * C_fa + (1 - Y_fa) * 0.7 * C_ac + Y_fa * C_bac)
  s_8 =  (-1 * C_va + (1 - Y_c4) * 0.54 * C_pro + (1 - Y_c4) * 0.31 * C_ac + Y_c4 * C_bac)
  s_9 =  (-1 * C_bu + (1 - Y_c4) * 0.8 * C_ac + Y_c4 * C_bac)
  s_10 =  (-1 * C_pro + (1 - Y_pro) * 0.57 * C_ac + Y_pro * C_bac)
  s_11 =  (-1 * C_ac + (1 - Y_ac) * C_ch4 + Y_ac * C_bac)
  s_12 =  ((1 - Y_h2) * C_ch4 + Y_h2 * C_bac)
  s_13 =  (-1 * C_bac + C_xc)

  Sigma =  (s_1 * Rho_1 + s_2 * Rho_2 + s_3 * Rho_3 + s_4 * Rho_4 + s_5 * Rho_5 + s_6 * Rho_6 + s_7 * Rho_7 + s_8 * Rho_8 + s_9 * Rho_9 + s_10 * Rho_10 + s_11 * Rho_11 + s_12 * Rho_12 + s_13 * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19))

  diff_S_IC = Q_ad / V_liq * (S_IC_in - S_IC) - Sigma - Rho_T_10
  ## eq10 ##


  diff_S_IN = Q_ad / V_liq * (S_IN_in - S_IN) - Y_su * N_bac * Rho_5 + (N_aa - Y_aa * N_bac) * Rho_6 - Y_fa * N_bac * Rho_7 - Y_c4 * N_bac * Rho_8 - Y_c4 * N_bac * Rho_9 - Y_pro * N_bac * Rho_10 - Y_ac * N_bac * Rho_11 - Y_h2 * N_bac * Rho_12 + (N_bac - N_xc) * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19) + (N_xc - f_xI_xc * N_I - f_sI_xc * N_I - f_pr_xc * N_aa) * Rho_1  # eq11

  diff_S_I = Q_ad / V_liq * (S_I_in - S_I) + f_sI_xc * Rho_1  # eq12


  # Differential equations 13 to 24 (particulate matter)

  diff_X_xc = Q_ad / V_liq * (X_xc_in - X_xc) - Rho_1 + Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19  # eq13

  diff_X_ch = Q_ad / V_liq * (X_ch_in - X_ch) + f_ch_xc * Rho_1 - Rho_2  # eq14

  diff_X_pr = Q_ad / V_liq * (X_pr_in - X_pr) + f_pr_xc * Rho_1 - Rho_3  # eq15

  diff_X_li = Q_ad / V_liq * (X_li_in - X_li) + f_li_xc * Rho_1 - Rho_4  # eq16

  diff_X_su = Q_ad / V_liq * (X_su_in - X_su) + Y_su * Rho_5 - Rho_13  # eq17

  diff_X_aa = Q_ad / V_liq * (X_aa_in - X_aa) + Y_aa * Rho_6 - Rho_14  # eq18

  diff_X_fa = Q_ad / V_liq * (X_fa_in - X_fa) + Y_fa * Rho_7 - Rho_15  # eq19

  diff_X_c4 = Q_ad / V_liq * (X_c4_in - X_c4) + Y_c4 * Rho_8 + Y_c4 * Rho_9 - Rho_16  # eq20

  diff_X_pro = Q_ad / V_liq * (X_pro_in - X_pro) + Y_pro * Rho_10 - Rho_17  # eq21

  diff_X_ac = Q_ad / V_liq * (X_ac_in - X_ac) + Y_ac * Rho_11 - Rho_18  # eq22

  diff_X_h2 = Q_ad / V_liq * (X_h2_in - X_h2) + Y_h2 * Rho_12 - Rho_19  # eq23

  diff_X_I = Q_ad / V_liq * (X_I_in - X_I) + f_xI_xc * Rho_1  # eq24

  # Differential equations 25 and 26 (cations and anions)
  diff_S_cation = Q_ad / V_liq * (S_cation_in - S_cation)  # eq25

  diff_S_anion = Q_ad / V_liq * (S_anion_in - S_anion)  # eq26

  if DAE_switch == 1 :

    diff_S_h2 = 0

    # Differential equations 27 to 32 (ion states, only for ODE implementation)
    diff_S_va_ion = 0  # eq27

    diff_S_bu_ion = 0  # eq28

    diff_S_pro_ion = 0  # eq29

    diff_S_ac_ion = 0  # eq30

    diff_S_hco3_ion = 0  # eq31

    diff_S_nh3 = 0  # eq32

  else:

    diff_S_h2 = Q_ad / V_liq * (S_h2_in - S_h2) + (1 - Y_su) * f_h2_su * Rho_5 + (1 - Y_aa) * f_h2_aa * Rho_6 + (1 - Y_fa) * 0.3 * Rho_7 + (1 - Y_c4) * 0.15 * Rho_8 + (1 - Y_c4) * 0.2 * Rho_9 + (1 - Y_pro) * 0.43 * Rho_10 - Rho_12 - Rho_T_8  # eq8

    # Differential equations 27 to 32 (ion states, only for ODE implementation)
    diff_S_va_ion = -Rho_A_4  # eq27

    diff_S_bu_ion = -Rho_A_5  # eq28

    diff_S_pro_ion = -Rho_A_6  # eq29

    diff_S_ac_ion = -Rho_A_7  # eq30

    diff_S_hco3_ion = -Rho_A_10  # eq31

    diff_S_nh3 = -Rho_A_11  # eq32

    phi =  (S_cation + S_nh4_ion - S_hco3_ion - (S_ac_ion / 64) - (S_pro_ion / 112) - (S_bu_ion / 160) - (S_va_ion / 208) - S_anion)

    S_H_ion = (-1 * phi / 2) + (0.5 * np.sqrt(phi ** 2 + 4 * K_w)) #this is just for ODE version


  # Gas phase equations: Differential equations 33 to 35
  diff_S_gas_h2 = (q_gas / V_gas * -1 * S_gas_h2) + (Rho_T_8 * V_liq / V_gas)  # eq33

  diff_S_gas_ch4 = (q_gas / V_gas * -1 * S_gas_ch4) + (Rho_T_9 * V_liq / V_gas)  # eq34

  diff_S_gas_co2 = (q_gas / V_gas * -1 * S_gas_co2) + (Rho_T_10 * V_liq / V_gas)  # eq35

  diff_S_H_ion = diff_S_co2 = diff_S_nh4_ion = 0 #to keep the output same length as input for ADM1_ODE funcion


  return diff_S_su, diff_S_aa, diff_S_fa, diff_S_va, diff_S_bu, diff_S_pro, diff_S_ac, diff_S_h2, diff_S_ch4, diff_S_IC, diff_S_IN, diff_S_I, diff_X_xc, diff_X_ch, diff_X_pr, diff_X_li, diff_X_su, diff_X_aa, diff_X_fa, diff_X_c4, diff_X_pro, diff_X_ac, diff_X_h2, diff_X_I, diff_S_cation, diff_S_anion, diff_S_H_ion, diff_S_va_ion,  diff_S_bu_ion, diff_S_pro_ion, diff_S_ac_ion, diff_S_hco3_ion, diff_S_co2,  diff_S_nh3, diff_S_nh4_ion, diff_S_gas_h2, diff_S_gas_ch4, diff_S_gas_co2


def simulate(t_step, solvermethod='RK45'):
  r = scipy.integrate.solve_ivp(ADM1_ODE, t_step, state_zero,method= solvermethod)
  return r.y

def DAESolve():
  global S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion, S_hco3_ion, S_nh3, S_H_ion, pH, p_gas_h2, S_h2
  
  ##  DAE calculations 
  eps = 0.0000001
  
  prevS_H_ion = S_H_ion
  

  #initial values for Newton-Raphson solver parameter
  shdelta = 1.0
  shgradeq = 1.0
  S_h2delta = 1.0
  S_h2gradeq = 1.0
  tol = 10 ** (-12) #solver accuracy tolerance
  maxIter = 1000 #maximum number of iterations for solver
  i = 1
  j = 1
  
  ## DAE solver for S_H_ion from Rosen et al. (2006)
  while ((shdelta > tol or shdelta < -tol) and (i <= maxIter)):
    S_va_ion = K_a_va * S_va / (K_a_va + S_H_ion)
    S_bu_ion = K_a_bu * S_bu / (K_a_bu + S_H_ion)
    S_pro_ion = K_a_pro * S_pro / (K_a_pro + S_H_ion)
    S_ac_ion = K_a_ac * S_ac / (K_a_ac + S_H_ion)
    S_hco3_ion = K_a_co2 * S_IC / (K_a_co2 + S_H_ion)
    S_nh3 = K_a_IN * S_IN / (K_a_IN + S_H_ion)
    shdelta = S_cation + (S_IN - S_nh3) + S_H_ion - S_hco3_ion - S_ac_ion / 64.0 - S_pro_ion / 112.0 - S_bu_ion / 160.0 - S_va_ion / 208.0 - K_w / S_H_ion - S_anion
    shgradeq = 1 + K_a_IN * S_IN / ((K_a_IN + S_H_ion) * (K_a_IN + S_H_ion)) + K_a_co2 * S_IC / ((K_a_co2 + S_H_ion) * (K_a_co2 + S_H_ion)) \
              + 1 / 64.0 * K_a_ac * S_ac / ((K_a_ac + S_H_ion) * (K_a_ac + S_H_ion)) \
              + 1 / 112.0 * K_a_pro * S_pro / ((K_a_pro + S_H_ion) * (K_a_pro + S_H_ion)) \
              + 1 / 160.0 * K_a_bu * S_bu / ((K_a_bu + S_H_ion) * (K_a_bu + S_H_ion)) \
              + 1 / 208.0 * K_a_va * S_va / ((K_a_va + S_H_ion) * (K_a_va + S_H_ion)) \
              + K_w / (S_H_ion * S_H_ion)
    S_H_ion = S_H_ion - shdelta / shgradeq
    if S_H_ion <= 0:
        S_H_ion = tol
    i+=1
  
  # pH calculation
  pH = - np.log10(S_H_ion)
  
  #DAE solver for S_h2 from Rosen et al. (2006) 
  while ((S_h2delta > tol or S_h2delta < -tol) and (j <= maxIter)):
    I_pH_aa = (K_pH_aa ** nn_aa) / (prevS_H_ion ** nn_aa + K_pH_aa ** nn_aa)
  
    I_pH_h2 = (K_pH_h2 ** n_h2) / (prevS_H_ion ** n_h2 + K_pH_h2 ** n_h2)
    I_IN_lim = 1 / (1 + (K_S_IN / S_IN))
    I_h2_fa = 1 / (1 + (S_h2 / K_I_h2_fa))
    I_h2_c4 = 1 / (1 + (S_h2 / K_I_h2_c4))
    I_h2_pro = 1 / (1 + (S_h2 / K_I_h2_pro))
  
    I_5 = I_pH_aa * I_IN_lim
    I_6 = I_5
    I_7 = I_pH_aa * I_IN_lim * I_h2_fa
    I_8 = I_pH_aa * I_IN_lim * I_h2_c4
    I_9 = I_8
    I_10 = I_pH_aa * I_IN_lim * I_h2_pro
  
    I_12 = I_pH_h2 * I_IN_lim
    Rho_5 = k_m_su * (S_su / (K_S_su + S_su)) * X_su * I_5  # Uptake of sugars
    Rho_6 = k_m_aa * (S_aa / (K_S_aa + S_aa)) * X_aa * I_6  # Uptake of amino-acids
    Rho_7 = k_m_fa * (S_fa / (K_S_fa + S_fa)) * X_fa * I_7  # Uptake of LCFA (long-chain fatty acids)
    Rho_8 = k_m_c4 * (S_va / (K_S_c4 + S_va)) * X_c4 * (S_va / (S_bu + S_va+ 1e-6)) * I_8  # Uptake of valerate
    Rho_9 = k_m_c4 * (S_bu / (K_S_c4 + S_bu)) * X_c4 * (S_bu / (S_bu + S_va+ 1e-6)) * I_9  # Uptake of butyrate
    Rho_10 = k_m_pro * (S_pro / (K_S_pro + S_pro)) * X_pro * I_10  # Uptake of propionate
    Rho_12 = k_m_h2 * (S_h2 / (K_S_h2 + S_h2)) * X_h2 * I_12  # Uptake of hydrogen
    p_gas_h2 = S_gas_h2 * R * T_ad / 16
    Rho_T_8 = k_L_a * (S_h2 - 16 * K_H_h2 * p_gas_h2)
    S_h2delta = Q_ad / V_liq * (S_h2_in - S_h2) + (1 - Y_su) * f_h2_su * Rho_5 + (1 - Y_aa) * f_h2_aa * Rho_6 + (1 - Y_fa) * 0.3 * Rho_7 + (1 - Y_c4) * 0.15 * Rho_8 + (1 - Y_c4) * 0.2 * Rho_9 + (1 - Y_pro) * 0.43 * Rho_10 - Rho_12 - Rho_T_8
    S_h2gradeq = - 1.0 / V_liq * Q_ad - 3.0 / 10.0 * (1 - Y_fa) * k_m_fa * S_fa / (K_S_fa + S_fa) * X_fa * I_pH_aa / (1 + K_S_IN / S_IN) / ((1 + S_h2 / K_I_h2_fa) * (1 + S_h2 / K_I_h2_fa)) / K_I_h2_fa - 3.0 / 20.0 * (1 - Y_c4) * k_m_c4 * S_va * S_va / (K_S_c4 + S_va) * X_c4 / (S_bu + S_va + eps) * I_pH_aa / (1 + K_S_IN / S_IN) / ((1 + S_h2 / K_I_h2_c4 ) * (1 + S_h2 / K_I_h2_c4 )) / K_I_h2_c4 - 1.0 / 5.0 * (1 - Y_c4) * k_m_c4 * S_bu * S_bu / (K_S_c4 + S_bu) * X_c4 / (S_bu + S_va + eps) * I_pH_aa / (1 + K_S_IN / S_IN) / ((1 + S_h2 / K_I_h2_c4 ) * (1 + S_h2 / K_I_h2_c4 )) / K_I_h2_c4 - 43.0 / 100.0 * (1 - Y_pro) * k_m_pro * S_pro / (K_S_pro + S_pro) * X_pro * I_pH_aa / (1 + K_S_IN / S_IN) / ((1 + S_h2 / K_I_h2_pro ) * (1 + S_h2 / K_I_h2_pro )) / K_I_h2_pro - k_m_h2 / (K_S_h2 + S_h2) * X_h2 * I_pH_h2 / (1 + K_S_IN / S_IN) + k_m_h2 * S_h2 / ((K_S_h2 + S_h2) * (K_S_h2 + S_h2)) * X_h2 * I_pH_h2 / (1 + K_S_IN / S_IN) - k_L_a
    S_h2 = S_h2 - S_h2delta / S_h2gradeq
    if S_h2 <= 0:
        S_h2 = tol
    j+=1

##time definition
days = 200
timeSteps = days * 24 * 4 #every 15 minutes 
t = np.linspace(0, days, timeSteps) #sequence of timesteps as fractions of days

#switch between ODE (0) and DAE (1) implementations
DAE_switch = 0

simulate_results = [0] * timeSteps #acts as a log for simulation results at each timestep

if DAE_switch == 0:
  solvermethod = 'Radau'
  tstep = t
  
  # solve ODE for next step 
  sim_S_su, sim_S_aa, sim_S_fa, sim_S_va, sim_S_bu, sim_S_pro, sim_S_ac, sim_S_h2, sim_S_ch4, sim_S_IC, sim_S_IN, sim_S_I, sim_X_xc, sim_X_ch, sim_X_pr, sim_X_li, sim_X_su, sim_X_aa, sim_X_fa, sim_X_c4, sim_X_pro, sim_X_ac, sim_X_h2, sim_X_I, sim_S_cation, sim_S_anion, sim_S_H_ion, sim_S_va_ion, sim_S_bu_ion, sim_S_pro_ion, sim_S_ac_ion, sim_S_hco3_ion, sim_S_co2, sim_S_nh3, sim_S_nh4_ion, sim_S_gas_h2, sim_S_gas_ch4, sim_S_gas_co2 = simulate(tstep, solvermethod)

  #store ODE simulation result states
  S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4, S_IC, S_IN, S_I, X_xc, X_ch, X_pr, X_li, X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2, X_I, S_cation, S_anion, S_H_ion, S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion, S_hco3_ion, S_co2, S_nh3, S_nh4_ion, S_gas_h2, S_gas_ch4, S_gas_co2 = \
  sim_S_su[-1], sim_S_aa[-1], sim_S_fa[-1], sim_S_va[-1], sim_S_bu[-1], sim_S_pro[-1], sim_S_ac[-1], sim_S_h2[-1], sim_S_ch4[-1], sim_S_IC[-1], sim_S_IN[-1], sim_S_I[-1], sim_X_xc[-1], sim_X_ch[-1], sim_X_pr[-1], sim_X_li[-1], sim_X_su[-1], sim_X_aa[-1], sim_X_fa[-1], sim_X_c4[-1], sim_X_pro[-1], sim_X_ac[-1], sim_X_h2[-1], sim_X_I[-1], sim_S_cation[-1], sim_S_anion[-1], sim_S_H_ion[-1], sim_S_va_ion[-1], sim_S_bu_ion[-1], sim_S_pro_ion[-1], sim_S_ac_ion[-1], sim_S_hco3_ion[-1], sim_S_co2[-1], sim_S_nh3[-1], sim_S_nh4_ion[-1], sim_S_gas_h2[-1], sim_S_gas_ch4[-1], sim_S_gas_co2[-1]
else:
  solvermethod = 'DOP853'
  
  for u in range(0,timeSteps-1):
  
    # span for next time step
    tstep = [t[u],t[u+1]]
  
    # solve ODE for next step 
    sim_S_su, sim_S_aa, sim_S_fa, sim_S_va, sim_S_bu, sim_S_pro, sim_S_ac, sim_S_h2, sim_S_ch4, sim_S_IC, sim_S_IN, sim_S_I, sim_X_xc, sim_X_ch, sim_X_pr, sim_X_li, sim_X_su, sim_X_aa, sim_X_fa, sim_X_c4, sim_X_pro, sim_X_ac, sim_X_h2, sim_X_I, sim_S_cation, sim_S_anion, sim_S_H_ion, sim_S_va_ion, sim_S_bu_ion, sim_S_pro_ion, sim_S_ac_ion, sim_S_hco3_ion, sim_S_co2, sim_S_nh3, sim_S_nh4_ion, sim_S_gas_h2, sim_S_gas_ch4, sim_S_gas_co2 = simulate(tstep, solvermethod)
  
    #store ODE simulation result states
    S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4, S_IC, S_IN, S_I, X_xc, X_ch, X_pr, X_li, X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2, X_I, S_cation, S_anion, S_H_ion, S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion, S_hco3_ion, S_co2, S_nh3, S_nh4_ion, S_gas_h2, S_gas_ch4, S_gas_co2 = \
    sim_S_su[-1], sim_S_aa[-1], sim_S_fa[-1], sim_S_va[-1], sim_S_bu[-1], sim_S_pro[-1], sim_S_ac[-1], sim_S_h2[-1], sim_S_ch4[-1], sim_S_IC[-1], sim_S_IN[-1], sim_S_I[-1], sim_X_xc[-1], sim_X_ch[-1], sim_X_pr[-1], sim_X_li[-1], sim_X_su[-1], sim_X_aa[-1], sim_X_fa[-1], sim_X_c4[-1], sim_X_pro[-1], sim_X_ac[-1], sim_X_h2[-1], sim_X_I[-1], sim_S_cation[-1], sim_S_anion[-1], sim_S_H_ion[-1], sim_S_va_ion[-1], sim_S_bu_ion[-1], sim_S_pro_ion[-1], sim_S_ac_ion[-1], sim_S_hco3_ion[-1], sim_S_co2[-1], sim_S_nh3[-1], sim_S_nh4_ion[-1], sim_S_gas_h2[-1], sim_S_gas_ch4[-1], sim_S_gas_co2[-1]
    DAESolve()
    state_zero = [S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4, S_IC, S_IN, S_I, X_xc, X_ch, X_pr, X_li, X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2, X_I, S_cation, S_anion, S_H_ion, S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion, S_hco3_ion, S_co2, S_nh3, S_nh4_ion, S_gas_h2, S_gas_ch4, S_gas_co2]
    simulate_results[u] = state_zero
      
S_nh4_ion =  (S_IN - S_nh3)
S_co2 =  (S_IC - S_hco3_ion)
pH = - np.log10(S_H_ion)

print('DAE_switch =', DAE_switch)

print ('S_su =', S_su, "\n",
         'S_aa =', S_aa, "\n",
         'S_fa =', S_fa, "\n",
         'S_va =', S_va, "\n",
         'S_bu =', S_bu, "\n",
         'S_pro =', S_pro, "\n",
         'S_ac =', S_ac, "\n",
         'S_h2 =', S_h2, "\n",
         'S_ch4 =', S_ch4, "\n",
         'S_IC =', S_IC, "\n",
         'S_IN =', S_IN, "\n",
         'S_I =', S_I, "\n",
         'X_xc =', X_xc, "\n",
         'X_ch =', X_ch, "\n",
         'X_pr =', X_pr, "\n",
         'X_li =', X_li, "\n",
         'X_su =', X_su, "\n",
         'X_aa =', X_aa, "\n",
         'X_fa =', X_fa, "\n",
         'X_c4 =', X_c4, "\n",
         'X_pro =', X_pro, "\n",
         'X_ac =', X_ac, "\n",
         'X_h2 =', X_h2, "\n",
         'X_I =', X_I, "\n",
         'S_cation =', S_cation,"\n", 
         'S_anion =', S_anion, "\n",
         'S_H_ion =', S_H_ion, "\n",
         'S_va_ion =', S_va_ion, "\n",
         'S_bu_ion =', S_bu_ion, "\n",
         'S_pro_ion =', S_pro_ion, "\n",
         'S_ac_ion =', S_ac_ion, "\n",
         'S_hco3_ion =', S_hco3_ion, "\n",
         'S_co2 =', S_co2, "\n",
         'S_nh3 =', S_nh3, "\n",
         'S_nh4_ion =', S_nh4_ion,"\n", 
         'S_gas_h2 =', S_gas_h2, "\n",
         'S_gas_ch4 =', S_gas_ch4, "\n",
         'S_gas_co2 =', S_gas_co2)


print(sim_S_pro)