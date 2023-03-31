#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <C:/Users/scott/OneDrive/Bureau/opencv/json.hpp>//télécharger le fichier json nlohmann (pas besoin d'installation, juste include ce fichier)

using json = nlohmann::json;






//utilisation de labelme pour générer les json
//pour les tests, quand on entoure les objets, if faut faire un rectangle avec le premier point en haut à gauche et le deuxième en bas à droite (sinon l'évaluation ne marche pas)



/**
* Rotation de l'image en fonction de lignes de Hough détéctées dans l'image
* 
* @param img l'image à modifier
* @param closing l'image binaire servant à trouver les lignes de Hough (qu'on modifie aussi)
*/
void hough_rotate(cv::Mat& img, cv::Mat& closing) {
	cv::Mat dst, lines;
	cv::Canny(closing, dst, 50, 200, 3);
	cv::HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0);

	double sum_theta = 0;
	int nb_values = 0;

	for (int i = 0; i < lines.rows; i++) {
		float t = lines.at<float>(cv::Point(1, i));
		if (t < 1) {
			sum_theta += t;
			nb_values += 1;
		}
	}
	double angle = 180 * (sum_theta / nb_values) / CV_PI;
	cv::Point2d center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
	cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
	cv::warpAffine(img, img, rotation_matrix, img.size());
	cv::warpAffine(closing, closing, rotation_matrix, closing.size());
}

/**
* Trie la liste de composantes connexes par rapport à l'aire (ordre décroissant) et séléction des n premiers éléments
* Pas de return, on modifie juste la matrice en paramètre
* 
* @param stats matrice contenant les composantes connexes, en particulier leurs coordonnées et aires
* @param b booléen qui indique si on veut séléctionner les n premiers éléments ou juste trier
* @param n entier qui indique le nombre de lignes de la matrice à récupérer
*/
void reverse_sort_by_fourth_column(cv::Mat& stats, bool b, int n) {
	cv::Mat col_4 = stats.col(4);
	cv::Mat1i idx;
	cv::sortIdx(col_4, idx, cv::SORT_EVERY_COLUMN + cv::SORT_DESCENDING);
	cv::Mat result(stats.rows, stats.cols, CV_32S);
	for (int y = 0; y < stats.rows; y++) {
		stats.row(idx(y)).copyTo(result.row(y));
	}
	if (b) {
		if (result.rows > n) {
			stats = result.rowRange(1, n);
		}
	}
	else {
		stats = result;
	}
}


//Fonction qui permet de trouver des minimums locaux dans un tableau
std::vector<int> find_local_minima(cv::Mat mat, int range) {
	// Ensure that the input mat is a 1xN matrix
	CV_Assert(mat.rows == 1);

	// Convert the input mat to a vector of integers
	std::vector<int> v;
	const unsigned char* mat_data = mat.ptr<unsigned char>();

	for (int i = 0; i < mat.cols; i++) {
		v.push_back(static_cast<int>(mat_data[i]));
	}

	std::vector<int> minima;

	// Loop through the vector and find local minima
	for (int i = range; i < v.size() - range; i++) {
		bool is_min = true;

		// Check if the current element is a local minimum
		for (int j = i - range; j <= i + range; j++) {
			if (v[j] < v[i]) {
				is_min = false;
				break;
			}
		}

		// If the current element is a local minimum, add it to the list of minima
		if (is_min) {
			minima.push_back(i);
		}
	}

	return minima;
}


//Fonction qui permet de tracer des lignes noires sur une image
//En fonction des minimums trouvés sur son histogramme projeté vertical
void set_values_to_zero(cv::Mat& mat, const std::vector<int>& indices) {
	
	for (int i = 0; i < indices.size(); i++) {
		int col_index = indices[i];
		if (col_index >= 0 && col_index < mat.cols) { // add bounds check
			for (int j = 0; j < mat.rows; j++) {
				mat.at<char>(j, col_index) = 0;
			}
		}
	}
}



/**
* segmentation du tableau, crop et segmentation du texte dans les tableaux
* @param path chemin de l'image du tableau
* @return un dictionnaire json qui contient deux clés "labels" et "coords" qui contiennent les listes des coordonnées et des labels
*/
json detection(std::string path) {

	//Conversion en hsv -> flou gaussien -> seuillage (sélection des pixels verts)
	cv::Mat hsv, gaussian, binary, closing, resized, img = cv::imread(path);
	cv::resize(img, resized, cv::Size(1000, 1000), cv::INTER_LINEAR);
	cv::cvtColor(resized, hsv, cv::COLOR_BGR2HSV);
	cv::GaussianBlur(hsv, gaussian, cv::Size(3, 3), 0);
	cv::inRange(gaussian, cv::Scalar(30, 0, 0), cv::Scalar(120, 255, 170), binary);

	//Fermeture binaire
	cv::Mat kernel = cv::Mat::ones(cv::Size(5, 5), CV_8UC1);
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::morphologyEx(binary, closing, cv::MORPH_CLOSE, kernel);

	//Recherche des contours, sélection du plus grand (le tableau)
	cv::findContours(closing, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	
	double largest_area = 0;
	int largest_area_index = 0;
	for (int i = 0; i < contours.size(); i++) {
		double area = cv::contourArea(contours[i]);
		if (area > largest_area) {
			largest_area = area;
			largest_area_index = i;
		}
	}

	//Approximation du polygone correspondant au contour trouvé par un quadrilatère (plus facile à gérer pour la suite)
	std::vector<cv::Point> approx;
	cv::approxPolyDP(contours[largest_area_index], approx, 0.1 * cv::arcLength(contours[largest_area_index], true), true);
	std::vector<std::vector<cv::Point>> contours_poly(1);
	contours_poly[0] = approx;
	drawContours(resized, contours_poly, 0, cv::Scalar(0, 0, 255), 2);

	cv::Point p0 = contours_poly[0][0];
	cv::Point p1 = contours_poly[0][1];
	cv::Point p2 = contours_poly[0][2];
	cv::Point p3 = contours_poly[0][3];

	int x0 = p0.x, y0 = p0.y;
	int x1 = p1.x, y1 = p1.y;
	int x2 = p2.x, y2 = p2.y;
	int x3 = p3.x, y3 = p3.y;

	int x_min = std::min({x0,x1,x2,x3});
	int y_min = std::min({y0,y1,y2,y3});
	int x_max = std::max({ x0,x1,x2,x3 });
	int y_max = std::max({ y0,y1,y2,y3 });

	cv::Mat mask = cv::Mat::zeros(resized.size(), CV_8UC1);
	cv::fillPoly(mask, contours_poly, cv::Scalar(255));
	cv::Mat outputImage = cv::Mat::zeros(resized.size(), resized.type());
	resized.copyTo(outputImage, mask);


	//Crop du tableau
	cv::Mat cropped_board = outputImage(cv::Range(y_min, y_max), cv::Range(x_min, x_max));


	//hough_rotate(img, closing); #remplacer par warp


	//Création de l'objet json qui va contenir tous les labels détectés:
	//"labels" : ["board", "board", "line", "line", "line", "word", "letter", etc]
	//"coords": [[xmin, ymin, w, h], [xmin, ymin, w, h], [xmin, ymin, w, h], etc]
	json js = {
		{"labels", json::array()},
		{"coords", json::array()}
	};
	js["coords"].insert(js["coords"].end(), json::array({ x_min, y_min, x_max, y_max}));
	js["labels"].insert(js["labels"].end(), "board");


	//Conversion du tableau croppé en gris, puis seuillage, puis dilatation binaire pour coller les mots qui sont sur la même ligne
	cv::Mat dilation_kernel_line = cv::Mat::ones(cv::Size(40, 1), CV_8UC1);//tester kernel pour aller que à gauche ou droite
	cv::Mat text_labels, text_stats, text_centroids, gray_board, binary_board, dilated_board;

	cv::cvtColor(cropped_board, gray_board, cv::COLOR_BGR2GRAY);
	cv::threshold(gray_board, binary_board, 180, 255, cv::THRESH_BINARY);
	cv::dilate(binary_board, dilated_board, dilation_kernel_line);//Tester fermeture binaire

	//Recherche des composantes connexes (lignes)
	cv::connectedComponentsWithStats(dilated_board, text_labels, text_stats, text_centroids);



	//reverse_sort_by_fourth_column(text_stats, true, 1);//appliquer seuillage ratio

	//Ajout des labels et coordonnées correspondantes aux lignes trouvées
	for (int k = 1; k < text_stats.rows; k++) {
		int x = text_stats.at<int>(cv::Point(0, k)) + x_min;
		int y = text_stats.at<int>(cv::Point(1, k)) + y_min;
		int w = text_stats.at<int>(cv::Point(2, k));
		int h = text_stats.at<int>(cv::Point(3, k));
		//if ((double)w / h <= 0.3 || (double)w / h >= 6) {
		cv::rectangle(resized, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);
		js["coords"].insert(js["coords"].end(), json::array({ x, y, x + w, y + h }));
		js["labels"].insert(js["labels"].end(), "line");
		//}

		//Détection des mots : crop des lignes, dilatation plus faibles pour trouver les mots (et pas les lignes)
		cv::Mat cropped_text = binary_board(cv::Rect(x - x_min, y - y_min, w, h)), cropped_closed;
		cv::Mat dilation_kernel_word = cv::Mat::ones(cv::Size(20, 1), CV_8UC1);
		cv::dilate(cropped_text, cropped_closed, dilation_kernel_word);

		//Recherche des coposantes connexes (les mots)
		cv::Mat word_labels, word_stats, word_centroids;
		cv::connectedComponentsWithStats(cropped_closed, word_labels, word_stats, word_centroids);

		//Ajout des labesl et coordonnées correspondantes aux mots trouvés
		for (int i = 1; i < word_stats.rows; i++) {
			int xw = word_stats.at<int>(cv::Point(0, i)) + x;
			int yw = word_stats.at<int>(cv::Point(1, i)) + y;
			int ww = word_stats.at<int>(cv::Point(2, i));
			int hw = word_stats.at<int>(cv::Point(3, i));
			//if ((double)w / h <= 0.3 || (double)w / h >= 6) {
			cv::rectangle(resized, cv::Rect(xw, yw, ww, hw), cv::Scalar(255, 0, 0), 1);
			js["coords"].insert(js["coords"].end(), json::array({ xw, yw, xw + ww, yw + hw }));
			js["labels"].insert(js["labels"].end(), "word");
			//}

			//Crop des mots trouvés pour traiter les lettres
			cv::Mat binary_cropped_word = cropped_text(cv::Rect(xw - x, yw - y, ww, hw));

			int width = binary_cropped_word.cols;
			int height = 1; // set height to 1

			//Calcul de l'histogramme projeté vertical
			cv::Mat hist = cv::Mat::zeros(height, width, CV_32FC1);
			cv::reduce(binary_cropped_word, hist, 0, cv::REDUCE_SUM, CV_32FC1);

			//Normalisation de l'histogramme (surtout pour la visualisation)
			double min_val, max_val;
			cv::minMaxLoc(hist, &min_val, &max_val);
			hist = hist / max_val * 255;
			hist.convertTo(hist, CV_8UC1);


			//Appel de la fonction permettant de trouver les minimums locaux
			//Param 1: histogramme
			//Param 2: fenêtre sur laquelle chercher chaque minimum
			std::vector<int> minima = find_local_minima(hist, 20);//voir avec flou gaussien
			/*
			for (int i = 0; i < minima.size(); i++) {
				std::cout << static_cast<int>(minima[i]) << " ";
			}*/

			
			//Pour chaque minimum trouvé, on trace une ligne noir dans l'image à l'indice correspondant, pour séparer les lettres
			set_values_to_zero(binary_cropped_word, minima);


			//Recherche des composantes connexes (les lettres)
			cv::Mat letter_labels, letter_stats, letter_centroids;
			cv::connectedComponentsWithStats(binary_cropped_word, letter_labels, letter_stats, letter_centroids);

			//Ajout des labesl et coordonnées correspondantes aux lettres trouvées
			for (int i = 1; i < letter_stats.rows; i++) {
				int xl = letter_stats.at<int>(cv::Point(0, i)) + xw;
				int yl = letter_stats.at<int>(cv::Point(1, i)) + yw;
				int wl = letter_stats.at<int>(cv::Point(2, i));
				int hl = letter_stats.at<int>(cv::Point(3, i));
				//if ((double)w / h <= 0.3 || (double)w / h >= 6) {
				cv::rectangle(resized, cv::Rect(xl, yl, wl, hl), cv::Scalar(0, 0, 0));
				js["coords"].insert(js["coords"].end(), json::array({ xl, yl, xl + wl, yl + hl }));
				js["labels"].insert(js["labels"].end(), "letter");
				//}

				//lettres
				//cv::Mat binary_cropped_letter = binary_cropped_word(cv::Rect(xl - xw, yl - yw, wl, hl));
				
			}
		}
	}

	cv::imshow("Image", resized);
	cv::waitKey(0);
	return js;
}
/**
* Simplifie le json généré par labelme donné en paramètre (enlève toutes les informations inutiles)
* @param path chemin du json
* @return nouveau json siomplifié
*/
json simplify_json(std::string path) {
	std::ifstream f(path);
	json js_read = json::parse(f);
	json js = {
		{ "labels",json::array() },
		{ "coords", json::array() }
	};
	for (int i = 0; i < js_read["shapes"].size(); i++) {
		js["labels"].insert(js["labels"].end(), js_read["shapes"][i]["label"]);
		json json_arr = json::array({
			static_cast<int>(js_read["shapes"][i]["points"][0][0] + 0.5),
			static_cast<int>(js_read["shapes"][i]["points"][0][1] + 0.5),
			static_cast<int>(js_read["shapes"][i]["points"][1][0] + 0.5),
			static_cast<int>(js_read["shapes"][i]["points"][1][1] + 0.5) });
		js["coords"].insert(js["coords"].end(), json_arr);
	}
	return js;
}

/**
* Calcule l'intersection sur l'union pour deux objets
* @param a liste contenant les coordonnées de l'objet a
* @param b liste contenant les coordonnées de l'objet b
* @return la valeur de l'iou au format double
**/
double intersection_over_union(json a, json b) {
	cv::Rect rect1((int)a[0], (int)a[1], (int)a[2] - (int)a[0], (int)a[3] - (int)a[1]);
	cv::Rect rect2((int)b[0], (int)b[1], (int)b[2] - (int)b[0], (int)b[3] - (int)b[1]);
	cv::Rect rect_intersect = rect1 & rect2;
	cv::Rect rect_union = rect1 | rect2;
	double inter_over_union = (double)rect_intersect.area() / (double)rect_union.area();
	return inter_over_union;
}


//Fonction qui retourne la le nombre de vrai positifs et l'iou moyenne pour chaque classe
std::tuple<int, double, int, double, int, double, int, double> true_positives(json g, json d, double iou_threshold) {
	double tp_board = 0, tp_board_mean_iou = 0, tp_line = 0, tp_line_mean_iou = 0, tp_word = 0, tp_word_mean_iou = 0, tp_letter = 0, tp_letter_mean_iou = 0;

	//parcours du json des vérités terrain
	for (int i = 0; i < g["labels"].size(); i++) {
		double best_iou = 0;
		int idx = 0;
		//parcours du json des objets détéctés par le programme
		for (int j = 0; j < d["labels"].size(); j++) {
			//Calcul de l'intersection sur l'union, su les labels sont les mêmes, l'iou et supérieur au seuil, on garde l'iou max
			double iou = intersection_over_union(g["coords"][i], d["coords"][j]);
			if (g["labels"][i] == d["labels"][j] && iou > iou_threshold && iou > best_iou) {
				best_iou = iou;
				idx = j;
			}
		}
		if (best_iou != 0) {
			if (g["labels"][i] == "board") {
				tp_board += 1;
				tp_board_mean_iou += best_iou;
			}
			if(g["labels"][i] == "line") {
				tp_line += 1;
				tp_line_mean_iou += best_iou;
			}
			if (g["labels"][i] == "word") {
				tp_word += 1;
				tp_word_mean_iou += best_iou;
			}
			if (g["labels"][i] == "letter") {
				tp_letter += 1;
				tp_letter_mean_iou += best_iou;
			}

		}
	}
	if (tp_board != 0) {
		tp_board_mean_iou /= tp_board;
	}
	if (tp_line != 0) {
		tp_line_mean_iou /= tp_line;
	}
	if (tp_word != 0) {
		tp_word_mean_iou /= tp_word;
	}
	if (tp_letter != 0) {
		tp_letter_mean_iou /= tp_letter;
	}

	return std::make_tuple(tp_board, tp_board_mean_iou, tp_line, tp_line_mean_iou, tp_word, tp_word_mean_iou, tp_letter, tp_letter_mean_iou);
}






//evalutation pour une seule image
void evaluation(json g, json d, double iou_threshold) {
	//precision = vp/(vp+fp)
	//recall = vp/(vp+fn
	
	auto tp = true_positives(g, d, iou_threshold);
	int tp_board = std::get<0>(tp), tp_line = std::get<2>(tp), tp_word = std::get<4>(tp), tp_letter = std::get<6>(tp);
	double tp_board_mean_iou = std::get<1>(tp), tp_line_mean_iou = std::get<3>(tp), tp_word_mean_iou = std::get<5>(tp), tp_letter_mean_iou = std::get<7>(tp);

	std::cout << "board" << std::endl;
	std::cout << tp_board << ", " << tp_board_mean_iou << std::endl;
	std::cout << "line" << std::endl;
	std::cout << tp_line << ", " << tp_line_mean_iou << std::endl;
	std::cout << "word" << std::endl;
	std::cout << tp_word << ", " << tp_word_mean_iou << std::endl;
	std::cout << "letter" << std::endl;
	std::cout << tp_letter << ", " << tp_letter_mean_iou << std::endl;

}



int main() {
	/*
	std::vector<cv::String> files;
	cv::glob("C:/Users/scott/onedrive/bureau/*.json", files);
	//std::string str = path.substr(0, str.size() - 4);
	//std::string path = "C:/Users/scott/OneDrive/Bureau/projet image m1/tab1.jpeg";
	*/


	std::string path = "C:/Users/scott/OneDrive/Bureau/images/18.jpg";
	std::string path1 = "C:/Users/scott/OneDrive/Bureau/images/18.json";

	json ground_truth = simplify_json(path1);
	//std::cout << ground_truth << std::endl;

	json detected = detection(path);
	//std::cout << detected << std::endl;

	evaluation(ground_truth, detected, 0.5);
	return 0;
}
