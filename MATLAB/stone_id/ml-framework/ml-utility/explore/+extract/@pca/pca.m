classdef pca < utility.AssignPVPairs & matlab.mixin.SetGet & matlab.mixin.CustomDisplay
    %PCA Principal Component Analysis (PCA) on raw data.
    %
    % Syntax
    %
    %   extract.pca( data )
    %   extract.pca( __, name, value, ... )
    %
    %   Algorithm      
    %       svd
    %       eig
    %       als
    %
    %   Normalization
    %       off
    %       zscore
    %       norm
    %       center
    %       scale
    %       range
    %
    % See also pca
    %
    % Copyright 2021 The MathWorks Inc.
    
    properties (SetAccess = private, GetAccess = public)     
        data      (:,:) double
    end
    
    
    properties       
        coeff
        score
        latent
        tsquared
        explained
        mu   
    end %public
    
    properties (SetObservable, AbortSet)
        Normalize       (1,1) string ...
            {mustBeMember(Normalize,["off", "zscore", "norm", "center", "scale", "range"])} = "zscore";
        Algorithm       (1,1) string ...
            {mustBeMember(Algorithm,["svd", "eig", "als"])} = "svd";
        Centered        (1,1) logical = true;
        Economy         (1,1) logical = true
        NumComponents   (1,1) double = 2
        Rows            (1,1) string ...
            {mustBeMember(Rows,["complete", "pairwise", "all"])} = "complete";
        Weights         double = []
        VariableWeights double = []
        Coeff0          double = []
        Score0          double = []
        Options         (1,1) struct =  ...
            statset('Display', 'off', 'MaxIter', 1000, 'TolFun', 1e-6, 'TolX', 1e-6 )
    end
    
    properties ( Access = private )
        data_ (:,:) double
        listeners_ (1,:) event.proplistener
    end
    
    
   
    
    methods        
        function obj = pca( data, varargin)
            %PCA Construct an instance of this class
            
            if nargin > 0
                obj.iParseInputArguments( data, varargin{:} );
                
                [~,remainArgs] = ...
                    utility.AssignPVPairs.splitArgs({'Normalize'}, varargin{:});
                
                if obj.Normalize ~= "off"
                    obj.data = normalize( obj.data_, obj.Normalize );
                else
                    obj.data = obj.data_;
                end %if obj.Normalize
                
                if isempty( remainArgs )
                    [obj.coeff, obj.score, obj.latent, ...
                        obj.tsquared, obj.explained, obj.mu ] = ...
                        pca( obj.data, "NumComponents", obj.NumComponents );
                else
                    [obj.coeff, obj.score, obj.latent, ...
                        obj.tsquared, obj.explained, obj.mu ] = ...
                        pca( obj.data, remainArgs{:} );
                end %if isempty(remainArgs)
                
                
                parameters = ["Normalize", "Algorithm", "Centered", "Economy",...
                    'NumComponents','Rows', 'Weights', 'VariableWeights',...
                    'Coeff0', 'Score0', 'Options'];
                
                for iParam = parameters
                   obj.listeners_= [ obj.listeners_,...
                       addlistener(obj,iParam,'PostSet',@(src,evt)obj.onPropertyChange(src,evt))];
                end
                
            end
            
        end %constructor

        function result = table( obj ) 
            %TABLE Return PCA scores as a table
            
            obj.NumComponents = size(obj.score, 2);

            result = array2table(obj.score,...
                "VariableNames", "Component" + (1:obj.NumComponents) );
            
        end %table
        
    end %public
    
    
    methods ( Access = private )   
        function iParseInputArguments(obj, data, varargin )
            
            if istable( data ) || istimetable( data )
                obj.data_ = data.Variables;
            else
                obj.data_ = data;
            end
            
            if ~isempty( varargin )
                set(obj, varargin{:})
            end
        end %iParseInputArguments 
        
        function onPropertyChange(obj, ~, ~)
            
            obj.apply()
            
        end %onPropertyChange
        
        
        function apply( obj )
            %APPLY Apply Configuration
            % 
            %   obj.apply()
            %
            
            if obj.Normalize ~= "off"
                obj.data = normalize( obj.data_, obj.Normalize );
            else
                obj.data = obj.data_;
            end %if obj.Normalize
            
            if obj.Algorithm == "als"
                parameters = {'Algorithm', 'Centered', 'Economy',...
                    'NumComponents','Rows', 'Weights', 'VariableWeights',...
                    'Coeff0', 'Score0', 'Options'};
            else
                parameters = {'Algorithm', 'Centered', 'Economy',...
                    'NumComponents','Rows', 'Weights', 'VariableWeights'};
            end %if obj.Algorithm
            
            values  = get(obj, parameters);
            toKeep  = cellfun(@(x)~isempty(x), values);
            args    = cell(1, numel( parameters(toKeep) )*2 );
            
            args( 1:2:end-1 ) = parameters( toKeep );
            args( 2:2:end)    = values( toKeep );
            
            [obj.coeff, obj.score, obj.latent, ...
                obj.tsquared, obj.explained, obj.mu ] = ...
                pca( obj.data, args{:} );
            
        end %apply method
        
    end %private methods
    
    
    methods (Access = protected)
        function  header = getHeader( obj )
            
            header =sprintf(['Principal Component Analysis: \n<a href = '...
                '"matlab: helpPopup %s">get help</a>\n'],class(obj));
            
        end
        
        
        function propGroup = getPropertyGroups(obj)
            
            propInfo = obj.getPropertyInfo();
            
            propGroup(1) = matlab.mixin.util.PropertyGroup(...
                cellstr(propInfo.DataProperties), "Data Properties");

            propGroup(2) = matlab.mixin.util.PropertyGroup(...
                cellstr(propInfo.InputProperties), "Configuration Properties");
            
            propGroup(3) = matlab.mixin.util.PropertyGroup(...
                cellstr(propInfo.OutputProperties), "Results Properties");
              
        end
   
        
        function value = getPropertyInfo( obj )
            
            value = struct();
            
            value.DataProperties =  ["Normalize","data"] ;
            
             if obj.Algorithm == "als" 
                 value.InputProperties = ["Algorithm", "Centered", "Economy",...
                     "NumComponents","Rows", "Weights", "VariableWeights",...
                     "Coeff0", "Score0", "Options"];
             else
                value.InputProperties = ["Algorithm", "Centered", "Economy",...
                    "NumComponents","Rows", "Weights", "VariableWeights"] ;   
             end
            
             value.OutputProperties = [ "coeff", "score", "latent", ...
                 "tsquared", "explained", "mu"];

        end %getPropertyInfo
        
        
        function footer = getFooter( obj )
            
            footer = sprintf(['\nSequence Learning Experiments: \n'...
                '<a href="matlab:methods(%s)">list all methods</a>\n'],class(obj));
            
            methodGroupString{1} = extract.pca.methodGroup( ...
                "apply", "Apply Methods" );
            
            methodGroupString{2} = extract.pca.methodGroup( ...
                ["pareto", "biplot"], "Visualization Methods" );
            
            footer = [footer methodGroupString{:}];
            
        end
    end %protected methods
    
    
    methods (Static) 
        function methodGroupString = methodGroup(methodList, someTitle )
            
            Title       = someTitle;
            NumMethods  = numel( methodList );
            MethodList  = methodList;
            
            
            groupTitle = sprintf('\n   %s\n', Title );
            
            methodListAsString = [];
            for iMethod = 1 : NumMethods
                thisString = sprintf('%35s: <a href="matlab: helpPopup %s/%s">help</a>\n',MethodList(iMethod),"lstm.Experiment",MethodList(iMethod));
                methodListAsString = [methodListAsString thisString]; %#ok<AGROW>
            end
            
            methodGroupString = [groupTitle methodListAsString];
            
        end %methodGroup            
    end %static [path management]  
end %classdef

