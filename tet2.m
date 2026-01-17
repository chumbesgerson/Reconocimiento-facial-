% ========== CONFIGURACIÓN MODIFICADA PARA TU ESTRUCTURA ==========
projectRoot = pwd; % Ruta del proyecto actual

% Verificar que tenemos las carpetas de emociones
classNames = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"];
foldersExist = true;
for i = 1:length(classNames)
    if ~exist(fullfile(projectRoot, char(classNames(i))), 'dir')
        foldersExist = false;
        disp(['✗ No se encontró carpeta: ' char(classNames(i))]);
    end
end

if foldersExist
    disp('✓ Dataset encontrado en formato de imágenes por emociones');
    skipCSVProcessing = true;
    
    % Configurar rutas
    resultsFolder = fullfile(projectRoot, 'resultados');
    if ~exist(resultsFolder, 'dir')
        mkdir(resultsFolder);
    end
    
    % Parámetros
    imageSize = [48 48];
    inputSize = [imageSize 1];
    
    % Omitir completamente el procesamiento de CSV
    outputImages = fullfile(projectRoot, 'dataset_organizado');
    
else
    error('No se encontraron las carpetas de emociones. Verifica la estructura.');
end

% Si el usuario quiere cargar un modelo, cargarlo
if choice == 1 && ~modeloCargado
    [file,path] = uigetfile('*.mat','Selecciona archivo .mat de modelo entrenado');
    if ~isequal(file,0)
        load(fullfile(path, file));
        modeloCargado = true;
    end
end

% Si no hay modelo cargado, necesitamos procesar y entrenar
if ~modeloCargado
    % ========== PASO 1: ORGANIZAR DATASET ==========
    disp('Paso 1: Organizando dataset de imágenes...');
    
    % Crear estructura estándar (Training 80%, Validation 10%, Test 10%)
    trainFolder = fullfile(outputImages, 'Training');
    valFolder = fullfile(outputImages, 'PublicTest');   % Validación
    testFolder = fullfile(outputImages, 'PrivateTest'); % Test
    
    % Crear carpetas
    for cls = classNames
        mkdir(fullfile(trainFolder, char(cls)));
        mkdir(fullfile(valFolder, char(cls)));
        mkdir(fullfile(testFolder, char(cls)));
    end
    
    % Distribuir imágenes (80-10-10)
    for clsIdx = 1:length(classNames)
        cls = classNames(clsIdx);
        sourceFolder = fullfile(projectRoot, char(cls));
        
        % Obtener todas las imágenes de esta emoción
        images = dir(fullfile(sourceFolder, '*.png'));
        if isempty(images)
            images = dir(fullfile(sourceFolder, '*.jpg'));
        end
        if isempty(images)
            images = dir(fullfile(sourceFolder, '*.jpeg'));
        end
        
        numImages = length(images);
        if numImages == 0
            warning(['No hay imágenes en: ' char(cls)]);
            continue;
        end
        
        % Mezclar aleatoriamente
        idx = randperm(numImages);
        
        % Calcular divisiones
        trainEnd = floor(0.8 * numImages);
        valEnd = floor(0.9 * numImages);
        
        % Copiar a Training (80%)
        for i = 1:trainEnd
            sourceFile = fullfile(sourceFolder, images(idx(i)).name);
            destFile = fullfile(trainFolder, char(cls), images(idx(i)).name);
            copyfile(sourceFile, destFile);
        end
        
        % Copiar a Validation (10%)
        for i = trainEnd+1:valEnd
            sourceFile = fullfile(sourceFolder, images(idx(i)).name);
            destFile = fullfile(valFolder, char(cls), images(idx(i)).name);
            copyfile(sourceFile, destFile);
        end
        
        % Copiar a Test (10%)
        for i = valEnd+1:numImages
            sourceFile = fullfile(sourceFolder, images(idx(i)).name);
            destFile = fullfile(testFolder, char(cls), images(idx(i)).name);
            copyfile(sourceFile, destFile);
        end
        
        fprintf('  %s: %d imágenes (Train: %d, Val: %d, Test: %d)\n', ...
            char(cls), numImages, trainEnd, valEnd-trainEnd, numImages-valEnd);
    end
    
    % ========== PASO 2: CREAR DATASTORES ==========
    disp('Creando imageDatastores...');
    trainDS = imageDatastore(trainFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    valDS = imageDatastore(valFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    testDS = imageDatastore(testFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    
    % Normalizar etiquetas
    trainDS.Labels = categorical(trainDS.Labels, cellstr(classNames));
    valDS.Labels = categorical(valDS.Labels, cellstr(classNames));
    testDS.Labels = categorical(testDS.Labels, cellstr(classNames));
    
    % Informar conteos
    disp('Conteo por clase en train:');
    disp(countEachLabel(trainDS));
    disp('Conteo por clase en val:');
    disp(countEachLabel(valDS));
    disp('Conteo por clase en test:');
    disp(countEachLabel(testDS));
    
%Paso 2 definir arquitectura CNN
disp('Paso 2 definir arquitectura CNN'); %Mensaje
layers = [
    imageInputLayer(inputSize, 'Name', 'entrada', 'Normalization', 'zerocenter') %Capa entrada
    convolution2dLayer(5, 32, 'Padding', 'same', 'Name', 'conv1') %Conv bloque 1
    batchNormalizationLayer('Name', 'bn1') %BatchNorm bloque 1
    reluLayer('Name', 'relu1') %ReLU bloque 1
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1') %MaxPool bloque 1
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2') %Conv bloque 2
    batchNormalizationLayer('Name', 'bn2') %BatchNorm bloque 2
    reluLayer('Name', 'relu2') %ReLU bloque 2
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2') %MaxPool bloque 2
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3') %Conv bloque 3
    batchNormalizationLayer('Name', 'bn3') %BatchNorm bloque 3
    reluLayer('Name', 'relu3') %ReLU bloque 3
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3') %MaxPool bloque 3
    fullyConnectedLayer(256, 'Name', 'fc_features') %Capa fully connected feature extractor
    reluLayer('Name', 'relu_fc') %ReLU de la capa fc_features
    dropoutLayer(0.5, 'Name', 'drop_fc') %Dropout para regularización
    fullyConnectedLayer(numel(classNames), 'Name', 'fc_final') %Capa final con num clases
    softmaxLayer('Name', 'softmax') %Softmax para probabilidades
    classificationLayer('Name', 'classoutput') %Capa de clasificación final
];
%Guardar arquitectura por si se requiere
save(fullfile(resultsFolder, 'cnn_architecture.mat'), 'layers', 'inputSize', 'classNames'); %Guardar arquitectura
%Mostrar gráfica de red opcional
%analyzeNetwork(layers) %Descomentar si se desea visualizar red
%Paso 3 entrenamiento CNN
disp('Paso 3 entrenamiento CNN'); %Mensaje
%Configurar augmentedImageDatastore con preprocessing de gris
augTrain = augmentedImageDatastore(imageSize, trainDS, 'ColorPreprocessing', 'gray2gray'); %ADS train
augVal   = augmentedImageDatastore(imageSize, valDS,   'ColorPreprocessing', 'gray2gray'); %ADS val
%Parámetros de entrenamiento
batchSize = 128; %Tamaño de mini batch
maxEpochs = 20; %Epocas de entrenamiento
initialLearnRate = 1e-3; %Tasa de aprendizaje inicial
%Determinar entorno de ejecución
if gpuDeviceCount > 0 %Si hay GPU
    executionEnv = 'gpu'; %Usar GPU
else
    executionEnv = 'auto'; %Dejar a MATLAB elegir
end
%Configurar opciones de entrenamiento
options = trainingOptions('adam', ...
    'InitialLearnRate', initialLearnRate, ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', batchSize, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', max(1, floor(numel(trainDS.Files)/batchSize)), ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', executionEnv);
%Entrenar la CNN
cnnNet = trainNetwork(augTrain, layers, options); %Entrenamiento principal
%Guardar CNN entrenada
save(fullfile(resultsFolder, 'cnn_entrenada.mat'), 'cnnNet', '-v7.3'); %Guardar modelo cnn
%Paso 4 extraer features con fc_features
disp('Paso 4 extrayendo features de fc_features'); %Mensaje
featureLayer = 'fc_features'; %Capa de la cual extraer features
%Función local para extraer features desde un datastore
function [featMat, labelVec] = extractFeaturesFromDS(network, ds, layerName, imSize)
    adsLocal = augmentedImageDatastore(imSize, ds, 'ColorPreprocessing', 'gray2gray'); %ADS local
    featMat = []; %Inicializar matriz de features
    labelVec = []; %Inicializar vector de etiquetas
    reset(adsLocal); %Reiniciar ADS
    while hasdata(adsLocal) %Mientras haya datos
        [imgs, info] = read(adsLocal); %Leer batch de imágenes
        acts = activations(network, imgs, layerName, 'OutputAs', 'rows'); %Obtener activations por filas
        featMat = [featMat; acts]; %Concatenar características
        for i = 1:numel(info.ImageIndex) %Recorrer indices de imagen
            labelVec = [labelVec; ds.Labels(info.ImageIndex(i))]; %Agregar etiqueta correspondiente
        end
    end
end
%Extraer features para train val test
[XtrainFeat, YtrainFeat] = extractFeaturesFromDS(cnnNet, trainDS, featureLayer, imageSize); %Features train
[XvalFeat,   YvalFeat]   = extractFeaturesFromDS(cnnNet, valDS,   featureLayer, imageSize); %Features val
[XtestFeat,  YtestFeat]  = extractFeaturesFromDS(cnnNet, testDS,  featureLayer, imageSize); %Features test
%Información de dimensiones
fprintf('Features train size %d x %d\n', size(XtrainFeat,1), size(XtrainFeat,2)); %Mostrar dimensiones
%Paso 5 entrenar MLP sobre features extraidas
disp('Paso 5 entrenamiento MLP sobre features'); %Mensaje
%Convertir etiquetas a indices numericos
YtrainIdx = grp2idx(YtrainFeat); %Indices train
%One hot encoding para patternnet
targetsTrain = full(ind2vec(YtrainIdx')); %Crear matriz one hot
%Configurar MLP con una capa oculta
hiddenUnits = 120; %Neurona oculta recomendada
mlp = patternnet(hiddenUnits); %Crear red MLP
mlp.trainFcn = 'trainscg'; %Algoritmo de entrenamiento
mlp.performFcn = 'crossentropy'; %Funcion de perdida
%Entrenar MLP con features como columnas
mlp = train(mlp, XtrainFeat', targetsTrain); %Entrenar MLP
%Definir wrapper predict para MLP
function preds = mlpPredictWrapper(localMLP, featMat, classes)
    y = localMLP(featMat'); %Obtener salida de la red
    [~, idx] = max(y, [], 1); %Elegir clase por maxima probabilidad
    idx = idx'; %Transponer indices
    preds = categorical(classes(idx)); %Convertir a categorical con nombres de clase
end
%Construir estructura del modelo final
mlpModel.net = mlp; %Red MLP interna
mlpModel.classNames = classNames; %Nombres de clase
mlpModel.predict = @(feat) mlpPredictWrapper(mlp, feat, classNames); %Funcion predict anonima
%Paso 6 evaluar y generar métricas
disp('Paso 6 evaluacion y metrics'); %Mensaje
%Predecir sobre conjunto de validacion y test
valPreds = mlpModel.predict(XvalFeat); %Predicciones val
testPreds = mlpModel.predict(XtestFeat); %Predicciones test
%Calcular accuracies
valAcc = mean(valPreds == YvalFeat); %Accuracy val
testAcc = mean(testPreds == YtestFeat); %Accuracy test
fprintf('Accuracy validacion %.4f\n', valAcc); %Mostrar val acc
fprintf('Accuracy test %.4f\n', testAcc); %Mostrar test acc
%Guardar precision por clase en archivo txt
classesAll = categories(YtestFeat); %Obtener categorias
fid = fopen(fullfile(resultsFolder, 'precision_por_clase.txt'), 'w'); %Abrir archivo texto
fprintf(fid, 'Clase\tPrecision\n'); %Encabezado
for i = 1:numel(classesAll) %Iterar por cada clase
    cls = classesAll{i}; %Clase actual
    idxs = YtestFeat == cls; %Indices de la clase
    prec = mean(testPreds(idxs) == YtestFeat(idxs)); %Calcular precision por clase
    fprintf(fid, '%s\t%.4f\n', cls, prec); %Escribir en archivo
end
fclose(fid); %Cerrar archivo
%Generar y guardar matriz de confusion como imagen
fig = figure('Visible', 'off'); %Crear figura oculta para guardar
cm = confusionchart(YtestFeat, testPreds, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized'); %Crear confusionchart
title('Matriz de confusion normalizada') %Titulo de figura
saveas(gcf, fullfile(resultsFolder, 'matriz_confusion.png')); %Guardar figura en carpeta resultados
close(gcf); %Cerrar figura
%Paso 7 guardar modelo final en .mat
disp('Guardando modelos en fichero .mat'); %Mensaje
save(fullfile(resultsFolder, 'modelo_entrenado.mat'), 'cnnNet', 'mlpModel', 'inputSize', 'classNames', '-v7.3'); %Guardar modelos y metadatos
disp('Modelos guardados en resultados/modelo_entrenado.mat'); %Notificar
%Paso 8 interfaz opcional en tiempo real con webcam
disp('Paso 8 interfaz de inferencia en tiempo real'); %Mensaje
%Intentar abrir camara y procesar frames
try
    camList = webcamlist; %Obtener lista de webcams
    if isempty(camList) %Si no hay camaras detectadas
        warning('No se detecto ninguna camara. Se omitira la interfaz en tiempo real'); %Advertencia
    else
        cam = webcam(1); %Abrir primera camara
        cam.Resolution = '640x480'; %Establecer resolucion de camara
        faceDetector = vision.CascadeObjectDetector('FrontalFaceCART'); %Detector de rostro
        hFig = figure('Name', 'Deteccion de emociones en tiempo real', 'NumberTitle', 'off'); %Crear figura
        hAx = axes('Parent', hFig); %Crear axes para mostrar imagen
        hImg = imshow(zeros(480,640,3,'uint8'), 'Parent', hAx); %Placeholder inicial
        disp('Presiona Ctrl+C en la ventana de comandos para detener la interfaz de tiempo real'); %Instruccion
        while ishandle(hFig) %Bucle principal mientras la figura exista
            frame = snapshot(cam); %Capturar frame de la camara
            gray = rgb2gray(frame); %Convertir frame a escala de grises
            bboxes = step(faceDetector, gray); %Detectar caras en el frame
            if ~isempty(bboxes) %Si se detectaron caras
                areas = bboxes(:,3) .* bboxes(:,4); %Calcular areas de bounding boxes
                [~, idxMax] = max(areas); %Seleccionar la cara mas grande
                bbox = bboxes(idxMax, :); %Caja seleccionada
                faceCrop = imcrop(frame, bbox); %Recortar rostro del frame original
                faceGray = imresize(rgb2gray(faceCrop), imageSize); %Convertir a gris y redimensionar
                faceSingle = im2single(faceGray); %Convertir a single para la red
                faceSingle = reshape(faceSingle, [imageSize 1]); %Asegurar forma HxWx1
                featReal = activations(cnnNet, faceSingle, featureLayer, 'OutputAs', 'rows'); %Extraer features en tiempo real
                predReal = mlpModel.predict(featReal); %Predecir usando el MLP entrenado
                frameOut = insertShape(frame, 'Rectangle', bbox, 'Color', 'green', 'LineWidth', 2); %Dibujar bbox
                frameOut = insertText(frameOut, [bbox(1), bbox(2)-20], char(predReal), 'FontSize', 18, 'BoxColor', 'black', 'TextColor', 'white'); %Agregar texto de etiqueta
            else
                frameOut = frame; %Si no hay caras, mostrar frame sin cambios
            end
            set(hImg, 'CData', frameOut); %Actualizar imagen en la figura
            drawnow; %Forzar renderizado
        end
        clear cam; %Liberar camara al cerrar la figura
    end
catch ME %Capturar errores de la interfaz
    warning('Error en la interfaz en tiempo real o camara no disponible'); %Advertencia amigable
    if exist('cam', 'var') %Si la variable cam existe
        clear cam; %Cerrar camara
    end
    rethrow(ME); %Re-lanzar el error para debug si se desea
end
%Fin del script
disp('Ejecucion del pipeline completada'); %Mensaje final