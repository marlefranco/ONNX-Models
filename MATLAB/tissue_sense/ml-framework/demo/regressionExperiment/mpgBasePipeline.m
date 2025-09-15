function value = mpgBasePipeline( data, options )

arguments
data table
options.Parameter
end

%Place your data preparation code here.


%Assign final dataset to result (e.g. value =...).
value = data;
featurenames = string( data.Properties.VariableNames(1:end-1) );
responsename = "MPG";

value = rmmissing(value);

%Partition
value = baseml.partition( value );

%Store Response, Features, and Partition
value = addprop(value, "Features", "table" );
value = addprop(value, "Response", "table" );
value = addprop(value, "TrainingObservations", "table" );

value.Properties.CustomProperties.Features = featurenames;
value.Properties.CustomProperties.Response = responsename;
value.Properties.CustomProperties.TrainingObservations = value.Partition == "Train";

end %function