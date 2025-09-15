classdef SemiSupervised < experiment.Base & experiment.mixin.SemiSupervised
    %SEMISUPERVISED Summary of this class goes here

    % Copyright 2021 The MathWorks Inc.
    
    properties
        Model (1,1) string ...
            {mustBeMember(Model,[...
            "automl"
            "graph"
            "self"]) } = "automl"
    end
    
    properties (Constant)
        Type = "SemiSupervised"
    end
  
    properties ( SetAccess = private, Hidden )
        ValidModelParameters = [
            "Learners"
            "CrossValidation"
            "KFold"
            "Holdout"
            "Seed"
            ] 
    end
    
    methods 
        function fit(obj)
        %FIT Train specified ml for items in items/configuration in the
        %experiment. Note the data preparation via obj.prepare() must occur
        %prior to fit.
        %
        % Syntax:
        %
        % obj.fit() train specified ml for all items in the experiment.
        %

        %Validate Model Parameters
        obj.validateModel();
        
        if ~isempty( obj.Trials )
            obj.resetmodels()
            model = obj.Model;
            switch model
                case {"automl"}
                    obj.( model );
                otherwise
                    obj.instanceml();
            end
        end %if ~isempty

        end %function
    end %methods
    
end %classdef 