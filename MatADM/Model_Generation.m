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
                       {'f_ch_xc'};
                       {'f_pr_xc'};
                       {'f_li_xc'};
                       {'f_xI_xc'};
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
for j=1:length(Model.Processes)
    for i=1:length(Model.Components)
    k=k+1    
    vijs(k)={strcat('v_',num2str(j),'_',num2str(i))};
    end
end


Model.Parameter_Names=[Model.Parameter_Names;
                        Cis';
                        vijs';
                        ]


Model.S=zeros(length(Model.Components),length(Model.Processes))
Model.Rates=zeros(length(Model.Processes),1)
Model.Parameter_Vals=zeros(size(Model.Parameter_Names))
Model.S(1,[12 13 14 15 16 24])=[Parameter_Vals(1) -1 Parameter_Vals(2) Parameter_Vals(3) Parameter_Vals(4) Parameter_Vals(5)];
Model.S(2,[1 14])=[1 -1];
Model.S(3,[2 15])=[1 -1];
Model.S(4,[1 3 16])=[1-Parameter_Vals(6) 1-Parameter_Vals(6) -1];
Model.S(5,[1 5 6 7 8 10 11 17])=[-1 (1-Parameter_Vals(7))*Parameter_Vals(8) (1-Parameter_Vals(7))*Parameter_Vals(9)  (1-Parameter_Vals(7))*Parameter_Vals(10) (1-Parameter_Vals(7))*Parameter_Vals(11) 0 -Parameter_Vals(7)*Parameter_Vals(11) Parameter_Vals(7)]
Model.S(5,10)= -sum(Parameter_Names([174,176:189]).*Parameter_Names([54,56:69]))