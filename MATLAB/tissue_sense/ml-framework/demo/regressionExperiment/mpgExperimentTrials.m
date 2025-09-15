%% Configure

load carbig

data = table(Acceleration,Displacement,Horsepower, ...
    Model_Year,Origin,Weight,MPG);


%% Generate new pipeline template 
newPipelineTemplate()

%% Default 

session = experiment.Regression("Data", data, ...
    "Model", "automl", ...
    "DataFcn", @(x,settings)mpgBasePipeline(x, settings{:}) );


session.validate()
session.build()
session.prepare()
session.fit()

%or session.run()


%% Model
modelParameter = optimizeParameter.new( ...
    "Name", "Learners", ...
    "Range", {["tree", "linear"]}, ...
    "Type", "Set");

session = experiment.Regression("Data", data, ...
    "Model", "automl", ...
    "DataFcn", @(x,settings)mpgBasePipeline(x, settings{:}), ...
    "ModelConfiguration", modelParameter);

session.validate()
session.build()
session.prepare()
session.fit()