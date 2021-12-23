#include "framework.h"
 
const char* vertexSource = R"(
	#version 330 
    precision highp float;
 
	uniform vec3 wLookAt, wRight, wUp;       
 
	layout(location = 0) in vec2 cCamWindowVertex;	
	out vec3 p;
 
	void main() {
		gl_Position = vec4(cCamWindowVertex, 0, 1);
		p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
	}
)";
 
const char* fragmentSource = R"(
	#version 330 
    precision highp float;
 
	struct Material {
		vec3 ka, kd, ks, F0;
		float  shininess;
		bool rough; 
	};
 
	struct PointLight{
		vec3 La, Le, position;
	};
 
	struct Hit {
		float t;
		vec3 position, normal;
		int mat;
		bool portal;	
	};
 
	struct Ray {
		vec3 start, dir;
	};
 
	#define M_PI 3.14159265
 
	const float epsilon = 0.0001f;
	const float wallDistance = 0.1f;
	const float scale = 1.0f;
	const int maxdepth = 5;
	const int dodekaSides = 12;
	const float innerRadius = 0.3f;
	const vec3 implicit = vec3(50.3f, 7.5f, 4.2f);
 
	uniform int planes[dodekaSides * 5];
	uniform vec3 wEye; 
	uniform vec3 v[20];
	uniform vec3 kd, ks, F0;
	uniform PointLight light;     
	uniform Material materials[2]; 
	
	in  vec3 p;					
	out vec4 fragmentColor;		
 
	vec4 TransformIntoQuat(const vec3 axis, const float angle) {
		vec4 q;
		q.x = axis.x * sin(angle * 0.5);
		q.y = axis.y * sin(angle * 0.5);
		q.z = axis.z * sin(angle * 0.5);
		q.w = cos(angle * 0.5);
		return q;
	}
 
	vec4 ConjugantQuat(vec4 q) {
		return vec4(-q.x, -q.y, -q.z, q.w);
	}
 
	vec4 MultQuat(const vec4 a, const vec4 b){
		vec4 q;
		q.x = (a.w * b.x) + (a.x * b.w) + (a.y * b.z) - (a.z * b.y);
		q.y = (a.w * b.y) - (a.x * b.z) + (a.y * b.w) + (a.z * b.x);
		q.z = (a.w * b.z) + (a.x * b.y) - (a.y * b.x) + (a.z * b.w);
		q.w = (a.w * b.w) - (a.x * b.x) - (a.y * b.y) - (a.z * b.z);
		return q;
	}
 
	vec3 RotatePosWithQuat(const vec3 position, const vec3 axis, const float angle){
		vec4 q = TransformIntoQuat(axis, angle);
		vec4 qConj = ConjugantQuat(q);
		vec4 qPos = vec4(position.x, position.y, position.z, 0);
 
		vec4 qTemp = MultQuat(q, qPos);
		q = MultQuat(qTemp, qConj);
 
		return vec3(q.x, q.y, q.z);
	}
 
	Hit ImplicitIntersect(const vec3 o, const Ray ray){
		Hit hit;
		hit.t = -1;
		hit.portal = false;
 
		float a = o.x * ray.dir.x * ray.dir.x + o.y * ray.dir.y * ray.dir.y;
		float b = 2.0f * (o.x * ray.start.x * ray.dir.x + o.y * ray.start.y * ray.dir.y) - o.z * ray.dir.z;
		float c = o.x * ray.start.x * ray.start.x + o.y * ray.start.y * ray.start.y - o.z * ray.start.z;
 
		if(a < epsilon)
			return hit;		
 
		float discr = b * b - 4 * a * c;
 
		if(discr < 0) 
			return hit;
 
		float sqrtDiscr = sqrt(discr);
		float t1 = (-b + sqrtDiscr) / (2.0f * a);
		float t2 = (-b - sqrtDiscr) / (2.0f * a);
 
		if(t1 <= 0) 
			return hit;
 
		vec3 O = vec3(0, 0, 0);
 
		vec3 tempPos1 = vec3(ray.start + ray.dir * t1);
		bool t1Inside = distance(tempPos1, O) <= innerRadius;
 
		vec3 tempPos2 = vec3(ray.start + ray.dir*t2);
		bool t2Inside = distance(tempPos2, O) <= innerRadius; 
 
		vec3 final;
 
		if(t1Inside && t2Inside){
			hit.t = (t2 > 0) ? t2 : t1;
			final = (t2 > 0) ? tempPos2 : tempPos1;
		}else if(t1Inside){
			hit.t = t1;
			final = tempPos1;
		}else if(t2Inside){
			hit.t = t2;
			final = tempPos2;
		}else 
			return hit;
 
		vec3 temp1 = vec3(1, 0, (2 * o.x * final.x) / o.z);
		vec3 temp2 = vec3(0, 1, (2 * o.y * final.y) / o.z);
 
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normalize(cross(temp1, temp2));
		hit.portal = false;
		hit.mat = 1;
 
		return hit;
	}
 
	void GetObjPlane(int i, out vec3 p, out vec3 normal){
		vec3 p1 = v[planes[5 * i] - 1], p2 = v[planes[5 * i + 1] - 1], p3 = v[planes[5 * i + 2] - 1];
		normal = cross(p2 - p1, p3 - p1);
		
		if(dot(p1, normal) < 0) 
			normal = -normal;
 
		p = p1 * scale + vec3(0, 0, epsilon);
	}
 
	float DistanceToWall(vec3 p, vec3 e1, vec3 e2){
		vec3 e1p = p - e1;
		vec3 e12 = e2 - e1;
		float cosTheta = dot(e1p, e12) / (length(e1p) * length(e12));
		float t = cosTheta * length(e1p);
		
		vec3 closest = e1 + normalize(e12) * t;
		vec3 pToClosest = closest - p;
		
		return length(pToClosest);
	}
	
	bool CloseToWall(vec3 p, int i){
		bool close = false;
		
		int k = 1;
		for(int j = 0; j < 5; ++j){
			vec3 e1 = v[planes[5 * i + j] - 1], e2 = v[planes[5 * i + k] - 1];
			float dist = DistanceToWall(p, e1, e2);
			
			if(dist <= wallDistance)
				close = true;
			k += 1;
			if(k == 5)
				k = 0;
		}
		return close;
	}
 
	Hit DodekaIntersect(Ray ray){
		Hit hit;
		hit.t = -1;
		hit.portal = false;
		
		for(int i = 0; i < dodekaSides; ++i){
			vec3 p1, normal;
			GetObjPlane(i, p1, normal);
 
			float ti = abs(dot(normal, ray.dir)) > epsilon ? dot(p1-ray.start, normal) / dot(normal, ray.dir) : -1;
			if(!(ti <= epsilon || (ti > hit.t && hit.t > 0))){
 
				vec3 pintersect = ray.start + ray.dir * ti;
				bool outside = false;
 
				for(int j = 0; j < dodekaSides; ++j){
					if(i != j) {
						vec3 p11, n;
						GetObjPlane(j, p11, n);
 
						if(dot(n, pintersect - p11) > 0){
							outside = true;
							break;
						}
					}
				}
 
				if(!outside){
					if(CloseToWall(pintersect, i)){
						hit.mat = 0;
						hit.portal = false;
					}else{
						hit.mat = 1;
						hit.portal = true;
					}
 
					hit.t = ti;
					hit.position = pintersect;
					hit.normal = normalize(normal);
				}
			}
		}
		return hit;
	}
 
	Hit FirstIntersect(Ray ray) {
		Hit bestHit;
		bestHit.t = -1;
		bestHit.portal = false;
 
		Hit hit = ImplicitIntersect(implicit, ray);
		if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))
				bestHit = hit;
 
		hit = DodekaIntersect(ray);
		if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))
				bestHit = hit;
		
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}
 
	vec3 Fresnel(vec3 F0, float cosTheta) { 
		return F0 + (vec3(1, 1, 1) - F0) * pow(1-cosTheta, 5);
	}
	
	vec3 Trace(in Ray ray) {
		vec3 weight = vec3(1, 1, 1);
		vec3 outRadiance = vec3(0, 0, 0);
 
		for(int d = 0; d < maxdepth + 1;) {
			Hit hit = FirstIntersect(ray);
 
			if (hit.t < 0) 
				return weight * light.La;
 
			if (materials[hit.mat].rough) {
				vec3 lightdir = normalize(light.position - hit.position);
 
				float cosTheta = dot(hit.normal, lightdir);
 
				if (cosTheta > 0) {
 
					outRadiance += weight * light.Le * materials[hit.mat].kd * cosTheta / pow(distance(hit.position, light.position), 2);
					vec3 halfway = normalize(-ray.dir + lightdir);
 
					float cosDelta = dot(hit.normal, halfway);
 
					if(cosDelta > 0) 
						outRadiance += weight * light.Le * materials[hit.mat].ks * pow(cosDelta, materials[hit.mat].shininess) / pow(distance(hit.position, light.position), 2);
				}
				d = 0;
				return outRadiance + weight * materials[0].ka * light.La; 
			}else{							
				if(hit.portal){
					ray.start = RotatePosWithQuat(hit.position + hit.normal * epsilon, hit.normal, 2 * M_PI / 5);
					ray.dir = reflect(ray.dir, hit.normal);
					ray.dir = RotatePosWithQuat(ray.dir, hit.normal, 2 * M_PI / 5);
					d++;
				}else{
					weight *= Fresnel(materials[hit.mat].F0, dot(-ray.dir, hit.normal));
					ray.start = hit.position + hit.normal * epsilon;
					ray.dir = reflect(ray.dir, hit.normal);			
				}
			}
		}
		return outRadiance + weight * light.La; 
	}
	
	void main() {
		Ray ray;
		ray.start = wEye; 
		ray.dir = normalize(p - wEye);
		fragmentColor = vec4(Trace(ray), 1); 
	}
)";
 
 
 
vec3 operator/(const vec3 a, const vec3 b) {
	return vec3(a.x / b.x, a.y / b.y, a.z / b.z);
}
 
vec3 NKappa(const vec3 n, const vec3 kappa) {
	vec3 one(1, 1, 1);
	return ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
}
 
struct Material {
	vec3 ka, kd, ks, F0;
	float  shininess;
	bool rough;
};
 
struct RoughMaterial : Material {
	RoughMaterial(const vec3 _kd, const vec3 _ks, const float _shininess) {
		ka = vec3(0.5f);
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
		rough = true;
	}
};
 
struct SmoothMaterial : Material {
	SmoothMaterial(const vec3 _F0) {
		F0 = _F0;
		rough = false;
	}
};
 
struct Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void Set(vec3 _eye, vec3 _La, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _La;
		fov = _fov;
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(vup, w)) * f * tanf(fov / 2);
		up = normalize(cross(w, right)) * f * tanf(fov / 2);
	}
	void Animate(const float dt) {
		eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.z - lookat.z) * sin(dt) + lookat.x,
			eye.y,
			-(eye.x - lookat.x) * sin(dt) + (eye.z - lookat.z) * cos(dt) + lookat.z);
		Set(eye, lookat, up, fov);
	}
};
 
struct Light {
	vec3 Le, La, position;
 
	Light(const vec3 le, const vec3 la, const vec3 pos) {
		Le = le; La = la;
		position = pos;
	}
};
 
class Shader : public GPUProgram {
public:
	void SetUniformMaterials(const std::vector<Material*>& materials) {
		char name[256];
		for (unsigned int mat = 0; mat < materials.size(); mat++) {
			sprintf(name, "materials[%d].ka", mat); setUniform(materials[mat]->ka, name);
			sprintf(name, "materials[%d].kd", mat); setUniform(materials[mat]->kd, name);
			sprintf(name, "materials[%d].ks", mat); setUniform(materials[mat]->ks, name);
			sprintf(name, "materials[%d].shininess", mat); setUniform(materials[mat]->shininess, name);
			sprintf(name, "materials[%d].F0", mat); setUniform(materials[mat]->F0, name);
			sprintf(name, "materials[%d].rough", mat); setUniform(materials[mat]->rough, name);
		}
	}
 
	void SetUniformLight(Light* light) {
		setUniform(light->La, "light.La");
		setUniform(light->Le, "light.Le");
		setUniform(light->position, "light.position");
	}
 
	void SetUniformCamera(const Camera& camera) {
		setUniform(camera.eye, "wEye");
		setUniform(camera.lookat, "wLookAt");
		setUniform(camera.right, "wRight");
		setUniform(camera.up, "wUp");
	}
};
 
 
class DodekaRoom {
	unsigned int vao;
	Camera camera;
	Light* light;
	std::vector<Material*> materials;
 
	const float g = 0.618f, G = 1.618f;
 
	const std::vector<int> planes = {
			1,2,16,5,13,
			1,13,9,10,14,
			1,14,6,15,2,
			2,15,11,12,16,
			3,4,18,8,17,
			3,17,12,11,20,
			3,20,7,19,4,
			19,10,9,18,4,
			16,12,17,8,5,
			5,8,18,9,13,
			14,10,19,7,6,
			6,7,20,11,15
	};
	
	const std::vector<vec3> v = {
		vec3(0,g,G), vec3(0,-g,G), vec3(0,-g,-G), vec3(0, g, -G), vec3(G, 0,g), vec3(-G, 0, g), vec3(-G, 0, -g), vec3(G, 0, -g), vec3(g,G,0), vec3(-g, G, 0), vec3(-g, -G, 0),
		vec3(g, -G, 0), vec3(1,1,1), vec3(-1,1,1), vec3(-1,-1,1), vec3(1, -1, 1), vec3(1, -1, -1), vec3(1,1,-1), vec3(-1, 1, -1), vec3(-1,-1,-1)
	};
public:
 
	void InitializeProjectTools() {
		vec3 eye(0.5f, 0.5f, 1);
		vec3 vup(0, 1, 0);
		vec3 lookat(0, 0, 0);
		float fov = 45 * (float)M_PI / 180;
		camera.Set(eye, lookat, vup, fov);
 
		light = new Light(vec3(3, 3, 3), vec3(0.4f, 0.3f, 0.3f), vec3(0.5f, 0.5f, -0.2f));
 
		vec3 kd(0.4f, 0.3f, 0.1f), ks(1, 1, 1);
		materials.push_back(new RoughMaterial(kd, ks, 500));
 
		vec3 n(0.17, 0.35, 1.5); vec3 kappa(3.1, 2.7, 1.9);
		materials.push_back(new SmoothMaterial(NKappa(n, kappa)));
	}
 
	void InitializeUniformObjects(Shader& shader) {
		for (int i = 0; i < v.size(); ++i)
			shader.setUniform(v[i], "v[" + std::to_string(i) + "]");
 
		for (int i = 0; i < planes.size(); ++i)
			shader.setUniform(planes[i], "planes[" + std::to_string(i) + "]");
 
		shader.SetUniformMaterials(materials);
		shader.SetUniformLight(light);
		shader.SetUniformCamera(camera);
	}
 
	void SetCamera(Shader& shader) {
		shader.SetUniformCamera(camera);
	}
 
	void Animate(const float dt) { 
		camera.Animate(dt); 
	}
 
	void InitializeOpenGLComponents() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
 
		unsigned int vbo;
		glGenBuffers(1, &vbo);
 
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
 
		glBufferData(GL_ARRAY_BUFFER, 
			sizeof(vertexCoords),
			vertexCoords,
			GL_STATIC_DRAW);
 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2,
			GL_FLOAT,
			GL_FALSE, 0, 
			NULL);
	}
 
	void DrawOnScreen() {
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
 
};
 
Shader shader;
DodekaRoom room;
 
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
 
	room.InitializeProjectTools();
	room.InitializeOpenGLComponents();
 
	shader.create(vertexSource, fragmentSource, "fragmentColor");
	shader.Use();
	room.InitializeUniformObjects(shader);
 
	for(int i = 0; i < 15;++i)
		room.Animate(0.1f);
}
 
void onDisplay() {
	glClearColor(1.0f, 0.5f, 0.8f, 1.0f);							
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
 
	room.SetCamera(shader);
	
	room.DrawOnScreen();
	glutSwapBuffers();								
}
 
void onKeyboard(unsigned char key, int pX, int pY) {
}
 
void onKeyboardUp(unsigned char key, int pX, int pY) {
}
 
void onMouse(int button, int state, int pX, int pY) {
}
 
void onMouseMotion(int pX, int pY) {
}
 
void onIdle() {
	room.Animate(0.01f);
	glutPostRedisplay();

}