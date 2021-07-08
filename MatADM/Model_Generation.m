Model.Processes=[
                    {'Disintegration'},
                    {'Hydrolysis carbohydrates'},
                    {'Hydrolysis of proteins'},
                    {'Hydrolysis of lipids'},
                    {'Uptake of sugars'},
                    {'Uptake of amino acids'},
                    {'Uptake of LCFA'},
                    {'Uptake of valerate'},
                    {'Uptake of butyrate'},
                    {'Uptake of propionate'},
                    {'Uptake of acetate'},
                    {'Uptake of Hydrogen'},
                    {'Decay of Xsu'},
                    {'Decay of Xaa'},
                    {'Decay of Xfa'},
                    {'Decay of Xc4'},
                    {'Decay of Xpro'},
                    {'Decay of Xac'},
                    {'Decay of Xh2'}
                    ];
Model.Components=[
                    {'Ssu'},
                    {'Saa'},
                    {'Sfa'},
                    {'Sva'},
                    {'Sbu'},
                    {'Spro'},
                    {'Sac'},
                    {'Sh2'},
                    {'Sch4'},
                    {'SIC'},
                    {'SIN'},
                    {'SI'},
                    {'Xc'},
                    {'Xch'},
                    {'Xpr'},
                    {'Xli'},
                    {'Xsu'},
                    {'Xaa'},
                    {'Xfa'},
                    {'Xc4'},
                    {'Xpro'},
                    {'Xac'},
                    {'Xh2'},
                    {'XI'}
                    ];
Model.Parameter_Names=[
                       {'f_sI_xc'};
                       {'f_a_li'};
                       {'Y_su'};
                       {'f_bu_su'};
                       {'f_pro_su'};
                       {'f_ac_su'};
                       {'f_h2_su'};
                       {'N_bac'};
                       {'Y_aa'};
                       {'Y_fa'};
                       {'Y_c4'};
                       {'Y_pro'};
                       {'Y_ac'};
                       {'Y_h2'};
                       {'Y_ch_xc'};
                       {'Y_pr_xc'};
                       {'Y_li_xc'};
                       {'f_xI_xc'};
                       {'Y_su'};
                       {'Y_aa'};
                       {'Y_fa'};
                       {'k_dis'};
                       {'k_hyd_ch'};
                       {'k_hyd_pr'};
                       {'k_hyd_li'};
                       {'k_m_su'};
                       {'k_m_aa'};
                       {'k_m_fa'};
                       {'k_m_c4'};
                       {'k_m_pr'};
                       {'k_m_ac'};
                       {'k_m_h2'};
                       {'K_s'};
                       {'k_dec_Xsu'};
                       {'k_dec_Xaa'};
                       {'k_dec_Xfa'};
                       {'k_dec_Xc4'};
                       {'k_dec_Xpro'};
                       {'k_dec_Xac'};
                       {'k_dec_Xh2'};
                       {'k_dis'};                      
                       ]
for i=1:length(Model.Components)
    Cis(i)={strcat('C',num2str(i))}
end
k=0
for i=1:length(Model.Components)
    for j=1:length(Model.Processes)
    k=k+1    
    vijs(k)={strcat('v_',num2str(i),'_',num2str(j))};
    end
end


Model.Parameter_Names=[Model.Parameter_Names;
                        Cis';
                        vijs';
                        ]


Model.S=zeros(length(Model.Components),length(Model.Processes))
Model.Rates=zeros(length(Model.Processes),1)

