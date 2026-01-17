% tet4_corregido.m - VERSIÓN CON ORDEN CORRECTO
% Todas las funciones al final del archivo

close all; clear; clc;

%% 1. OPCIÓN DE CARGAR MODELO
disp('¿Deseas cargar un modelo entrenado en .mat?');
choice = input('Escribe 1 para cargar modelo, 0 para entrenar desde cero: ');

cargarModelo = false;
modeloCargado = false;
if choice == 1
    [file,path] = uigetfile('*.mat','Selecciona archivo .mat de modelo entrenado');
    if isequal(file,0)
        disp('No se seleccionó ningún archivo. Se entrenará desde cero');
    else
        disp(['Cargando modelo desde: ', fullfile(path,file)]);
        load(fullfile(path, file));
        cargarModelo = true;
        modeloCargado = true;
        disp('✓ Modelo cargado exitosamente');
    end
end

%% 2. CONFIGURACIÓN PARA TU ESTRUCTURA
projectRoot = pwd;
classNames = ["angry","disgust","fear","happy","sad","surprise","neutral"];
imageSize = [48 48];
inputSize = [imageSize 1];

% Verificar que existan las carpetas de emociones
disp('=== VERIFICANDO ESTRUCTURA ===');
foldersOK = true;
for i = 1:length(classNames)
    folderPath = fullfile(projectRoot, classNames(i));
    if ~exist(folderPath, 'dir')
        foldersOK = false;
        disp(['✗ Falta carpeta: ' classNames(i)]);
    else
        imgCount = length(dir(fullfile(folderPath, '*.png'))) + ...
                  length(dir(fullfile(folderPath, '*.jpg'))) + ...
                  length(dir(fullfile(folderPath, '*.jpeg')));
        disp(['  ' classNames{i} ': ' num2str(imgCount) ' imágenes']);
    end
end

if ~foldersOK
    error('Faltan carpetas de emociones. Asegúrate de tener: angry, disgust, fear, happy, sad, surprise, neutral');
end

resultsFolder = fullfile(projectRoot, 'resultados');
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

%% 3. SI NO HAY MODELO CARGADO, ENTRENAR
if ~modeloCargado
    disp('=== INICIANDO ENTRENAMIENTO DESDE CERO ===');
    
    % 3.1 Organizar dataset (80-10-10)
    disp('Paso 1: Organizando dataset...');
    organizedData = fullfile(projectRoot, 'dataset_organizado');
    
    if exist(organizedData, 'dir')
        rmdir(organizedData, 's');
    end
    
    % Crear estructura
    trainFolder = fullfile(organizedData, 'Training');
    valFolder = fullfile(organizedData, 'PublicTest');
    testFolder = fullfile(organizedData, 'PrivateTest');
    
    for cls = classNames
        mkdir(fullfile(trainFolder, char(cls)));
        mkdir(fullfile(valFolder, char(cls)));
        mkdir(fullfile(testFolder, char(cls)));
    end
    
    % Distribuir imágenes
    totalImages = 0;
    for clsIdx = 1:length(classNames)
        cls = classNames(clsIdx);
        sourceFolder = fullfile(projectRoot, char(cls));
        
        % Obtener imágenes
        images = dir(fullfile(sourceFolder, '*.png'));
        if isempty(images)
            images = dir(fullfile(sourceFolder, '*.jpg'));
        end
        if isempty(images)
            images = dir(fullfile(sourceFolder, '*.jpeg'));
        end
        
        numImages = length(images);
        totalImages = totalImages + numImages;
        
        if numImages == 0
            warning(['No hay imágenes en: ' char(cls)]);
            continue;
        end
        
        % Mezclar
        idx = randperm(numImages);
        trainEnd = floor(0.8 * numImages);
        valEnd = floor(0.9 * numImages);
        
        % Copiar imágenes
        for i = 1:trainEnd
            sourceFile = fullfile(sourceFolder, images(idx(i)).name);
            destFile = fullfile(trainFolder, char(cls), images(idx(i)).name);
            copyfile(sourceFile, destFile);
        end
        
        for i = trainEnd+1:valEnd
            sourceFile = fullfile(sourceFolder, images(idx(i)).name);
            destFile = fullfile(valFolder, char(cls), images(idx(i)).name);
            copyfile(sourceFile, destFile);
        end
        
        for i = valEnd+1:numImages
            sourceFile = fullfile(sourceFolder, images(idx(i)).name);
            destFile = fullfile(testFolder, char(cls), images(idx(i)).name);
            copyfile(sourceFile, destFile);
        end
        
        fprintf('  %s: %d imágenes\n', char(cls), numImages);
    end
    
    fprintf('Total imágenes procesadas: %d\n', totalImages);
    
    % 3.2 Crear datastores
    disp('Paso 2: Creando datastores...');
    trainDS = imageDatastore(trainFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    valDS = imageDatastore(valFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    testDS = imageDatastore(testFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    
    trainDS.Labels = categorical(trainDS.Labels, cellstr(classNames));
    valDS.Labels = categorical(valDS.Labels, cellstr(classNames));
    testDS.Labels = categorical(testDS.Labels, cellstr(classNames));
    
    % 3.3 Mostrar conteos
    disp('Conteo por clase (Training):');
    disp(countEachLabel(trainDS));
    
    % 3.4 Definir arquitectura CNN
    disp('Paso 3: Definiendo arquitectura CNN...');
    layers = [
        imageInputLayer(inputSize, 'Name', 'entrada', 'Normalization', 'zerocenter')
        convolution2dLayer(5, 32, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
        batchNormalizationLayer('Name', 'bn3')
        reluLayer('Name', 'relu3')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')
        fullyConnectedLayer(256, 'Name', 'fc_features')
        reluLayer('Name', 'relu_fc')
        dropoutLayer(0.5, 'Name', 'drop_fc')
        fullyConnectedLayer(numel(classNames), 'Name', 'fc_final')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'classoutput')
    ];
    
    % 3.5 Entrenar CNN
    disp('Paso 4: Entrenando CNN...');
    augTrain = augmentedImageDatastore(imageSize, trainDS, 'ColorPreprocessing', 'gray2gray');
    augVal = augmentedImageDatastore(imageSize, valDS, 'ColorPreprocessing', 'gray2gray');
    
    options = trainingOptions('adam', ...
        'InitialLearnRate', 1e-3, ...
        'MaxEpochs', 20, ...
        'MiniBatchSize', 64, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', augVal, ...
        'ValidationFrequency', 30, ...
        'Verbose', true, ...
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', 'auto');
    
    cnnNet = trainNetwork(augTrain, layers, options);
    save(fullfile(resultsFolder, 'cnn_entrenada.mat'), 'cnnNet', '-v7.3');
    
    % 3.6 Extraer features
    disp('Paso 5: Extrayendo features...');
    featureLayer = 'fc_features';
    
    % Usar función local (definida al final)
    [XtrainFeat, YtrainFeat] = extractFeaturesFromDS(cnnNet, trainDS, featureLayer, imageSize);
    [XvalFeat, YvalFeat] = extractFeaturesFromDS(cnnNet, valDS, featureLayer, imageSize);
    [XtestFeat, YtestFeat] = extractFeaturesFromDS(cnnNet, testDS, featureLayer, imageSize);
    
    % 3.7 Entrenar MLP
    disp('Paso 6: Entrenando MLP...');
    YtrainIdx = grp2idx(YtrainFeat);
    targetsTrain = full(ind2vec(YtrainIdx'));
    
    mlp = patternnet(120);
    mlp.trainFcn = 'trainscg';
    mlp.performFcn = 'crossentropy';
    mlp = train(mlp, XtrainFeat', targetsTrain);
    
    % Crear estructura del modelo
    mlpModel.net = mlp;
    mlpModel.classNames = classNames;
    mlpModel.predict = @(feat) mlpPredictWrapper(mlp, feat, classNames);
    
    % 3.8 Evaluar
    disp('Paso 7: Evaluando modelo...');
    valPreds = mlpModel.predict(XvalFeat);
    testPreds = mlpModel.predict(XtestFeat);
    
    valAcc = mean(valPreds == YvalFeat);
    testAcc = mean(testPreds == YtestFeat);
    
    fprintf('✓ Accuracy validación: %.2f%%\n', valAcc*100);
    fprintf('✓ Accuracy test: %.2f%%\n', testAcc*100);
    
    % Guardar modelo final
    save(fullfile(resultsFolder, 'modelo_entrenado.mat'), ...
         'cnnNet', 'mlpModel', 'inputSize', 'classNames', 'imageSize', '-v7.3');
    disp('✓ Modelo guardado en resultados/modelo_entrenado.mat');
    
else
    disp('✓ Usando modelo cargado. Saltando entrenamiento.');
    
    % Verificar que el modelo tenga las variables necesarias
    if ~exist('imageSize', 'var')
        imageSize = [48 48];
        inputSize = [imageSize 1];
    end
    if ~exist('classNames', 'var')
        classNames = ["angry","disgust","fear","happy","sad","surprise","neutral"];
    end
end

%% 4. INTERFAZ DE CÁMARA - VERSIÓN ROBUSTA
disp('=== INTERFAZ DE CÁMARA ===');

% Verificar cámaras disponibles
try
    camList = webcamlist;
    if isempty(camList)
        error('No se detectaron cámaras');
    end
    disp(['Cámara detectada: ' camList{1}]);
catch
    camList = {'Integrated Camera'}; % Asumir que existe
    disp('Usando cámara por defecto: Integrated Camera');
end

% Variables globales para las funciones callback
stopCamera = false;
frameOut = [];
cam = [];
hFig = [];
fpsText = [];

try
    % Intentar inicializar la cámara
    cam = [];
    maxAttempts = 3;
    
    for attempt = 1:maxAttempts
        try
            fprintf('Intento %d de %d...\n', attempt, maxAttempts);
            
            if attempt == 1
                cam = webcam(1);
                cam.Resolution = '640x480';
            elseif attempt == 2
                cam = webcam(camList{1});
                cam.Resolution = '320x240';
            elseif attempt == 3
                cam = webcam(1);
            end
            
            testFrame = snapshot(cam);
            fprintf('✓ Cámara inicializada correctamente\n');
            fprintf('  Tamaño de frame: %d x %d\n', size(testFrame, 1), size(testFrame, 2));
            break;
            
        catch ME
            fprintf('  Intento fallido: %s\n', ME.message);
            if ~isempty(cam)
                clear cam;
                cam = [];
            end
            pause(1);
        end
    end
    
    if isempty(cam)
        error('No se pudo inicializar la cámara después de %d intentos', maxAttempts);
    end
    
    % ========== CONFIGURAR INTERFAZ GRÁFICA ==========
    
    % Detector de rostros
    faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');
    
    % Crear figura
    hFig = figure('Name', 'Detección de Emociones - Tiempo Real', ...
                 'NumberTitle', 'off', ...
                 'Position', [100, 100, 800, 650], ...
                 'MenuBar', 'none', ...
                 'ToolBar', 'none');
    
    % Área para mostrar video
    hAx = axes('Parent', hFig, 'Units', 'normalized', ...
              'Position', [0.05, 0.15, 0.9, 0.75]);
    hImg = imshow(zeros(480, 640, 3, 'uint8'), 'Parent', hAx);
    title(hAx, 'Detección de Emociones Faciales', 'FontSize', 14, 'FontWeight', 'bold');
    
    % Panel de control
    uicontrol('Style', 'text', ...
             'String', 'Estado: ACTIVO', ...
             'Position', [20, 20, 150, 30], ...
             'FontSize', 11, ...
             'FontWeight', 'bold', ...
             'BackgroundColor', [0.2, 0.8, 0.2], ...
             'ForegroundColor', [1, 1, 1]);
    
    uicontrol('Style', 'text', ...
             'String', 'Presiona ESC para salir', ...
             'Position', [200, 20, 200, 30], ...
             'FontSize', 10, ...
             'BackgroundColor', [0.8, 0.8, 0.8]);
    
    % Contador de FPS
    fpsText = uicontrol('Style', 'text', ...
                       'String', 'FPS: --', ...
                       'Position', [450, 20, 100, 30], ...
                       'FontSize', 10);
    
    % Botón para capturar foto
    uicontrol('Style', 'pushbutton', ...
             'String', 'Capturar Foto', ...
             'Position', [600, 20, 120, 30], ...
             'FontSize', 10, ...
             'Callback', @capturePhotoCallback);
    
    % Configurar callbacks
    set(hFig, 'CloseRequestFcn', @closeAppCallback);
    set(hFig, 'KeyPressFcn', @keyPressCallback);
    
    % Variables de control
    stopCamera = false;
    frameCount = 0;
    tStart = tic;
    photoCount = 0;
    
    % ========== BUCLE PRINCIPAL ==========
    disp('Iniciando detección en tiempo real...');
    disp('Instrucciones:');
    disp('  1. Mira hacia la cámara');
    disp('  2. Asegúrate de tener buena iluminación');
    disp('  3. Presiona ESC para salir');
    disp('  4. Botón "Capturar Foto" para guardar imágenes');
    
    while ~stopCamera && ishandle(hFig)
        try
            % Capturar frame
            frame = snapshot(cam);
            frameCount = frameCount + 1;
            
            % Convertir a escala de grises para detección
            grayFrame = rgb2gray(frame);
            
            % Detectar rostros
            bboxes = faceDetector(grayFrame);
            
            % Procesar cada rostro detectado
            if ~isempty(bboxes)
                % Ordenar por área (de mayor a menor)
                areas = bboxes(:,3) .* bboxes(:,4);
                [~, sortIdx] = sort(areas, 'descend');
                
                % Tomar hasta 2 rostros (el más prominente)
                numFaces = min(2, size(bboxes, 1));
                frameOut = frame;
                
                for f = 1:numFaces
                    bbox = bboxes(sortIdx(f), :);
                    
                    % Preprocesar rostro
                    faceCrop = imcrop(frame, bbox);
                    faceGray = imresize(rgb2gray(faceCrop), imageSize);
                    faceSingle = im2single(faceGray);
                    faceSingle = reshape(faceSingle, [imageSize 1]);
                    
                    % Predecir emoción
                    featReal = activations(cnnNet, faceSingle, 'fc_features', 'OutputAs', 'rows');
                    predReal = mlpModel.predict(featReal);
                    
                    % Color según emoción
                    emotionColors = containers.Map(...
                        {'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'}, ...
                        {[255,0,0], [128,0,128], [128,128,0], [0,255,0], [0,0,255], [255,165,0], [128,128,128]});
                    
                    if isKey(emotionColors, char(predReal))
                        boxColor = emotionColors(char(predReal)) / 255;
                    else
                        boxColor = [0, 1, 0]; % Verde por defecto
                    end
                    
                    % Dibujar bounding box
                    frameOut = insertShape(frameOut, 'Rectangle', bbox, ...
                                          'Color', boxColor, ...
                                          'LineWidth', 3, ...
                                          'Opacity', 0.7);
                    
                    % Agregar texto con la emoción
                    textPos = [bbox(1), bbox(2)-35];
                    frameOut = insertText(frameOut, textPos, ...
                                         char(predReal), ...
                                         'FontSize', 14, ...
                                         'BoxColor', boxColor, ...
                                         'TextColor', 'white', ...
                                         'BoxOpacity', 0.8);
                end
            else
                frameOut = frame;
                
                % Mostrar mensaje "Buscando rostros..."
                [h, w, ~] = size(frame);
                frameOut = insertText(frameOut, [w/2-100, 50], ...
                                     'Buscando rostros...', ...
                                     'FontSize', 16, ...
                                     'BoxColor', 'black', ...
                                     'TextColor', 'white', ...
                                     'AnchorPoint', 'Center');
            end
            
            % Calcular FPS
            elapsedTime = toc(tStart);
            if elapsedTime > 1
                fps = frameCount / elapsedTime;
                set(fpsText, 'String', sprintf('FPS: %.1f', fps));
                frameCount = 0;
                tStart = tic;
            end
            
            % Actualizar display
            set(hImg, 'CData', frameOut);
            drawnow limitrate;
            
        catch ME
            warning('Error en procesamiento de frame: %s', ME.message);
            if ~ishandle(hFig)
                break;
            end
        end
    end
    
    % ========== LIMPIAR RECURSOS ==========
    disp('Cerrando aplicación...');
    
    if ~isempty(cam) && isvalid(cam)
        clear cam;
        disp('Cámara liberada');
    end
    
    if ishandle(hFig)
        close(hFig);
    end
    
    disp('Interfaz cerrada correctamente');
    
catch ME
    % ========== MODO DE PRUEBA ALTERNATIVO ==========
    warning('Error en interfaz principal: %s', ME.message);
    disp('Iniciando modo de prueba alternativa...');
    
    % Probar con una imagen estática
    testWithStaticImage(classNames, projectRoot, cnnNet, mlpModel, imageSize);
end

disp('=== EJECUCIÓN COMPLETADA ===');

%% ========== TODAS LAS FUNCIONES VAN AQUÍ (AL FINAL) ==========

% Función para cerrar la aplicación
function closeAppCallback(~, ~)
    global stopCamera cam;
    stopCamera = true;
    if ~isempty(cam) && isvalid(cam)
        clear cam;
        disp('Cámara liberada desde callback');
    end
    delete(gcf);
end

% Callback para capturar foto
function capturePhotoCallback(~, ~)
    global photoCount frameOut projectRoot;
    
    if isempty(photoCount)
        photoCount = 0;
    end
    photoCount = photoCount + 1;
    
    % Guardar frame actual
    photoName = sprintf('captura_emocion_%03d.jpg', photoCount);
    
    % Asegurar que resultados existe
    if ~exist('resultsFolder', 'var')
        resultsFolder = fullfile(pwd, 'resultados');
    end
    if ~exist(resultsFolder, 'dir')
        mkdir(resultsFolder);
    end
    
    imwrite(frameOut, fullfile(resultsFolder, photoName));
    fprintf('✓ Foto guardada: %s\n', photoName);
    
    % Mostrar mensaje
    msgbox(sprintf('Foto guardada como:\n%s', photoName), 'Captura Exitosa');
end

% Callback para tecla ESC
function keyPressCallback(~, event)
    global stopCamera;
    if strcmp(event.Key, 'escape')
        stopCamera = true;
        disp('Tecla ESC presionada. Cerrando...');
    end
end

% Función para extraer features
function [featMat, labelVec] = extractFeaturesFromDS(network, ds, layerName, imSize)
    adsLocal = augmentedImageDatastore(imSize, ds, 'ColorPreprocessing', 'gray2gray');
    featMat = [];
    labelVec = [];
    reset(adsLocal);
    
    while hasdata(adsLocal)
        [imgs, info] = read(adsLocal);
        acts = activations(network, imgs, layerName, 'OutputAs', 'rows');
        featMat = [featMat; acts];
        
        for i = 1:numel(info.ImageIndex)
            labelVec = [labelVec; ds.Labels(info.ImageIndex(i))];
        end
    end
end

% Función wrapper para MLP
function preds = mlpPredictWrapper(localMLP, featMat, classes)
    y = localMLP(featMat');
    [~, idx] = max(y, [], 1);
    idx = idx';
    preds = categorical(classes(idx));
end

% Función para prueba con imagen estática
function testWithStaticImage(classNames, projectRoot, cnnNet, mlpModel, imageSize)
    disp(' MODO DE PRUEBA CON IMAGEN ESTÁTICA ');
    
    % Buscar cualquier imagen en las carpetas de emociones
    for emotion = classNames
        folderPath = fullfile(projectRoot, char(emotion));
        if exist(folderPath, 'dir')
            imageFiles = dir(fullfile(folderPath, '*.jpg'));
            if isempty(imageFiles)
                imageFiles = dir(fullfile(folderPath, '*.png'));
            end
            
            if ~isempty(imageFiles)
                % Tomar una imagen aleatoria
                imgIdx = randi(length(imageFiles));
                imgPath = fullfile(folderPath, imageFiles(imgIdx).name);
                
                fprintf('Probando con imagen: %s\n', imageFiles(imgIdx).name);
                
                img = imread(imgPath);
                
                % Detectar rostros
                faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');
                grayImg = rgb2gray(img);
                bboxes = faceDetector(grayImg);
                
                if ~isempty(bboxes)
                    % Procesar cada rostro
                    for i = 1:size(bboxes, 1)
                        bbox = bboxes(i, :);
                        
                        % Preprocesar
                        faceCrop = imcrop(img, bbox);
                        faceGray = imresize(rgb2gray(faceCrop), imageSize);
                        faceSingle = im2single(faceGray);
                        faceSingle = reshape(faceSingle, [imageSize 1]);
                        
                        % Predecir
                        featReal = activations(cnnNet, faceSingle, 'fc_features', 'OutputAs', 'rows');
                        predReal = mlpModel.predict(featReal);
                        
                        % Mostrar resultado
                        figure('Name', 'Prueba de Reconocimiento', ...
                              'NumberTitle', 'off', ...
                              'Position', [200, 200, 800, 600]);
                        
                        imshow(img);
                        hold on;
                        
                        % Dibujar bounding box
                        rectangle('Position', bbox, ...
                                 'EdgeColor', [0, 1, 0], ...
                                 'LineWidth', 3);
                        
                        % Agregar texto
                        text(bbox(1), bbox(2)-25, ...
                             ['Emoción: ' char(predReal)], ...
                             'Color', 'white', ...
                             'BackgroundColor', 'black', ...
                             'FontSize', 14, ...
                             'FontWeight', 'bold');
                        
                        title(sprintf('Imagen: %s - Emoción detectada: %s', ...
                              imageFiles(imgIdx).name, char(predReal)), ...
                              'FontSize', 16);
                        
                        fprintf('✓ Emoción detectada: %s\n', char(predReal));
                        fprintf('  Presiona cualquier tecla para continuar...\n');
                        pause;
                    end
                    break;
                end
            end
        end
    end
end