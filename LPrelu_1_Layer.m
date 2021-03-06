classdef LPrelu_1_Layer < nnet.layer.Layer
    properties (Learnable)
        Cut
        Alpha
    end
    methods
        function layer = LPrelu_1_Layer(numChannels, name)
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "LPrelu_1_Layer with " + numChannels + " channels";
            
            layer.Cut = 10;
            layer.Alpha = .05;
        end
        
        function Z = predict(layer, X)
            Z =   (0 < X & X <= layer.Cut).*X + (layer.Cut < X).*(layer.Cut + 0.05.*(X - layer.Cut));
        end
        
    end
end
