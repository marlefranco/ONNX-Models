function value = basePipelineCluster( tbl, options )
% basePipelineCluster Prepare/process data for cluster training
%

    arguments
       tbl table
       options.Normalization (1,1) string {mustBeMember(options.Normalization, ...
           ["zscore", "center", "range", "scale", "norm"])} = "zscore"
    end
    
    %Normalize
    tbl  = normalize(tbl, options.Normalization);

    %Place your data preparation code here.
    value = tbl;
    
    featurenames = value.Properties.VariableNames;
    
    %Store Features and Pipeline options.
    value = addprop(value, "Features", "table" );
    value = addprop(value, "Options", "table" );
    
    value.Properties.CustomProperties.Features = featurenames;
    value.Properties.CustomProperties.Options = options;
    
             
end %basePipelineCluster