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
                       {'f_sI_xc'},
                       {'f_a_li'},
                       {'Y_su'},
                       {'f_bu_su'},
                       {'f_pro_su'},
                       {'f_ac_su'},
                       {'f_h2_su'},
                       {'N_bac'},
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

Model.S=zeros(length(Model.Components),length(Model.Processes))
Model.Rates=zeros(length(Model.Processes),1)
