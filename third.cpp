#include "framework.h"
 
const int tessellationLevel = 100;
const float radius = 0.07f;
const vec3  G = vec3(0, 0, 1) * 9.81f;
const float deathLimit = 0.001f;
 
template<class T> struct Dnum {
	float f;
	T d; 
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};
 
template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Tan(Dnum<T>  g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f) * g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f) * g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
	return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}
 
typedef Dnum<vec2> Dnum2;
 
struct Material {
	vec3 kd, ks, ka;
	float shininess;
	bool rubberSheet;
 
	Material(vec3 _kd, vec3 _ks, vec3 _ka, float shine = 30, bool rubber = false) {
		kd = _kd;
		ks = _ks;
		ka = _ka;
		shininess = shine;
		rubberSheet = rubber;
	}
};
 
struct Light {
	vec3 La, Le;
	vec4 wLightPos, beginningPosition, otherBeginningPositon;
 
	Light(vec3 la, vec3 le, vec4 begin, vec4 other) {
		La = la; 
		Le = le;
		wLightPos = begin;
		beginningPosition = begin;
		otherBeginningPositon = other;
	}
 
	vec4 MultQuat(const vec4 a, const vec4 b) {
		vec4 q;
		q.x = (a.w * b.x) + (a.x * b.w) + (a.y * b.z) - (a.z * b.y);
		q.y = (a.w * b.y) - (a.x * b.z) + (a.y * b.w) + (a.z * b.x);
		q.z = (a.w * b.z) + (a.x * b.y) - (a.y * b.x) + (a.z * b.w);
		q.w = (a.w * b.w) - (a.x * b.x) - (a.y * b.y) - (a.z * b.z);
		return q;
	}
 
	void Animate(const float t) {
		vec4 q(cosf(t / 4), sinf(t / 4) * cosf(t) / 2, sinf(t / 4) * sinf(t) / 2, sinf(t / 4) * sqrtf(3 / 4));
		vec4 p = beginningPosition - otherBeginningPositon;
 
		wLightPos = otherBeginningPositon + MultQuat(q, p);
	}
};
 
struct RenderState {
	mat4 MVP, M, Minv, V, P;
	Material* material;
	std::vector<Light> lights;
	Texture* texture;
	vec3 wEye;
};
 
class Shader : public GPUProgram {
public:
	virtual void Bind(RenderState state) = 0;
 
	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
		setUniform(material.rubberSheet, name + ".rubberSheet");
	}
 
	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};
 
class PhongShader : public Shader {
	const char* vertexSource = R"(
		#version 330
		precision highp float;
 
		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
 
		uniform mat4  MVP, M, Minv; 
		uniform Light[8] lights;    
		uniform int   nLights;
		uniform vec3  wEye;         
 
		layout(location = 0) in vec3  vtxPos;            
		layout(location = 1) in vec3  vtxNorm;      	 
 
		out vec3 wNormal;		 
		out vec3 wView;       
		out vec3 wLight[8];		 
		out vec3 sheetCoord;
 
		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; 
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
			sheetCoord = vtxPos;
		}
	)";
 
	const char* fragmentSource = R"(
		#version 330
		precision highp float;
 
		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
 
		struct Material {
			vec3 kd, ks, ka;
			float shininess;
			bool rubberSheet;
		};
 
		uniform Material material;
		uniform int nLights;
		uniform Light[8] lights;  
 
		in vec3 wNormal;     
		in vec3 wView;        
		in vec3 wLight[8];     
		in vec3 sheetCoord;
 
		const int levels = 30;
		
        out vec4 fragmentColor; 
 
		void main() {
			float level = floor(sheetCoord.z * levels) / levels + 1;
			bool levelEnabled = sheetCoord.z < 0 ? true : false;
 
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
 
			if (dot(N, vec3(0,0,1)) < 0) 
				N = -N;	
 
			vec3 ka = material.ka;
			vec3 kd = material.kd;
			vec3 ks = material.ks;
			bool rubberSheet = material.rubberSheet;
 
			if(levelEnabled && rubberSheet)
				kd *= level;
 
			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 La = lights[i].La;
				vec3 Le = lights[i].Le;
 
				if(levelEnabled && rubberSheet)
					La *= level;
 
				vec3 L = normalize(lights[i].wLightPos.xyz - sheetCoord);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				
				vec3 pos = lights[i].wLightPos.xyz;
				float d = pow(distance(sheetCoord, pos), 2);				
 
				radiance += ka * La + kd * cost * Le / d;
 
				radiance += ks * pow(cosd, material.shininess) * Le / d;		
			}	
 
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }
 
	void Bind(RenderState state) {
		Use(); 	
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniformMaterial(*state.material, "material");
 
		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};
 
class Geometry {
protected:
	unsigned int vao, vbo;       
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};
 
class ParamSurface : public Geometry {
	struct VertexData {
		vec3 position, normal;
		vec2 texcoord;
	};
 
	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }
 
	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;
 
	VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		vtxData.texcoord = vec2(u, v);
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vtxData.normal = cross(drdU, drdV);
		return vtxData;
	}
 
	void create(int N = tessellationLevel, int M = tessellationLevel) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;	
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);  
		glEnableVertexAttribArray(1);  
 
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
	}
 
	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};
 
class RubberSheet : public ParamSurface {
	std::vector<vec3> heavyObjects;
	float mass;
public:
 
	RubberSheet(std::vector<vec3> HO, float _mass) {
		heavyObjects = HO;
		mass = _mass;
		create();
	}
 
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2 - 1;
		V = V * 2 - 1;
		X = U;
		Y = V;
 
		Dnum2 nZ;
 
		for (int i = 0; i < heavyObjects.size(); ++i) {
			vec3 o = heavyObjects[i];
			nZ = nZ + Dnum2(o.z) *Pow((Pow(Pow(U - Dnum2(o.x), 2) + Pow(V - Dnum2(o.y), 2), 0.5f) + 0.005f),-1);
		}
		Z = nZ * -1;
	}
 
	void addHeavyObject(vec2 hObj) {
		vec3 obj(hObj.x, hObj.y, mass);
		mass += 0.5f;
		heavyObjects.push_back(obj);
		create();
	}
 
	std::vector<vec3> getHeavyObjects() {
		return heavyObjects;
	}
 
	float getMass() {
		return mass;
	}
 
	bool HitHeavyObject(vec2 pos) {
		for (vec3 heavyObj : heavyObjects) {
			vec2 temp = vec2(heavyObj.x, heavyObj.y) - pos;
 
			if (length(temp) < 1 / heavyObj.z * deathLimit)//deathLimit)
				return true;
		}
		return false;
	}
};
 
class Sphere : public ParamSurface {
public:
	Sphere() { 
		create(); 
	}
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
		X = Cos(U) * Sin(V); Y = Sin(U) * Sin(V); Z = Cos(V);
	}
};
 
struct Object {
	Shader* shader;
	Material* material;
	Geometry* geometry;
	vec3 scale, translation, rotationAxis, bottomPoint, velocity;
	float rotationAngle;
	bool hasCamera = false;
	bool alive = true;
	bool hasMovedAlready = false;
public:
	Object(Shader* _shader, Material* _material, Geometry* _geometry, vec3 _bottomPoint = vec3(0,0,0)) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0), bottomPoint(_bottomPoint) {
		shader = _shader;
		material = _material;
		geometry = _geometry;
	}
 
	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}
 
	vec3 MeasureNormal(Dnum2& X, Dnum2& Y, Dnum2& Z) {
		vec3 position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vec3 normal = cross(drdU, drdV);
		return normal;
	}
 
	void PutToSurface(RubberSheet* rubberSheet) {
		Dnum2 u = Dnum2((bottomPoint.x + 1) / 2, vec2(1, 0));
		Dnum2 v = Dnum2((bottomPoint.y + 1) / 2, vec2(0, 1));
		Dnum2 x, y, z;
 
		rubberSheet->eval(u, v, x, y, z);
		vec3 normal = normalize(MeasureNormal(x, y, z));
		translation = vec3(x.f, y.f, z.f) + normal * radius;
	}
 
	void ApplyForce(RubberSheet* rubberSheet, float dt) {
		Dnum2 u = Dnum2((bottomPoint.x + 1) / 2, vec2(1, 0));
		Dnum2 v = Dnum2((bottomPoint.y + 1) / 2, vec2(0, 1));
		Dnum2 x, y, z;
		
		rubberSheet->eval(u, v, x, y, z);
 
		vec3 normal = MeasureNormal(x, y, z);
		normal.z = 1;
		normal = normalize(normal);
 
		float proj = dot(normal, G);
		vec3 projVec = proj * normal;
		vec3 force = G - projVec;
 
		velocity = velocity + force * dt;
	}
 
	void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		shader->Bind(state);
		geometry->Draw();
	}
};
 
struct Camera {
	vec3 wEye, wLookat, wVup;
	Object* currSphere;
 
	virtual mat4 V() = 0;
	virtual mat4 P() = 0;
};
 
struct PerspCamera : public Camera { 
	float fov, asp, fp, bp;		
public:
	PerspCamera() {
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 0.001f; bp = 20;
	}
	mat4 V() override { 
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}
 
	mat4 P() override {
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp * bp / (bp - fp), 0);
	}
};
 
struct OrtoCamera : public Camera {
	const float n = 0, f = 100, w = 2, h = 2;
 
	OrtoCamera() {}
 
	mat4 V() override {
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}
 
	mat4 P() override {
		return mat4(2 / w, 0, 0, 0,
			0, 2 / h, 0, 0,
			0, 0, -2 / (f - n), 0,
			0, 0, -(f + n) / (f - n), 1);
	}
};
 
enum Quarter {
	UPPERRIGHT, BOTTOMRIGHT, UPPERLEFT, BOTTOMLEFT
};
 
class Scene {
	Object* sheet;
	Camera* camera;
	std::vector<Light> lights;
	std::vector<Object*> spheres;
	Shader* phongShader;
	Object* currentSphere;
	RubberSheet* rubberSheet;
 
	void createLights() {
		vec3 La(0.4f, 0.4f, 0.4f);
		vec3 Le(13, 13, 13);
		vec4 pos1(-1.5, 1.5, 5, 0); 
		vec4 pos2(1.5, -1.5, 5, 0);
 
		Light l1(La, Le, pos1, pos2);
		Light l2(La, Le, pos2, pos1);
 
		lights.push_back(l1);
		lights.push_back(l2);
	}
 
	void createCamera() {
		camera->wEye = vec3(0, 0, 1);
		camera->wLookat = vec3(0, 0, 0);
		camera->wVup = vec3(0, 1, 0);
	}
 
	void createSheet(std::vector<vec3> HO, float mass = 0.015f) {
		vec3 kd(1, 0.703f, 0);
		vec3 ks(0.5f, 0.5f, 0.5f);
		vec3 ka(0.1f, 0.1f, 0.1f);
		
		Material* mat = new Material(kd, ks, ka, 50, true);
 
		rubberSheet = new RubberSheet(HO, mass);
 
		Object* sheetObject = new Object(phongShader, mat, rubberSheet);
		sheet = sheetObject;
	}
 
public:
	bool persp = false;
 
	void addHeavyObject(vec2 hObj) {
		std::vector<vec3> prevHO = rubberSheet->getHeavyObjects();
		float mass = rubberSheet->getMass();
		vec3 newHeavy(hObj.x, hObj.y, mass);
 
		Quarter quarter;
		if (!(newHeavy.x == 0 && newHeavy.y == 0)) {
			if (newHeavy.x < 0 && newHeavy.y < 0)
				quarter = BOTTOMLEFT;
			else if (newHeavy.x < 0 && newHeavy.y > 0)
				quarter = UPPERLEFT;
			else if (newHeavy.x > 0 && newHeavy.y < 0)
				quarter = BOTTOMRIGHT;
			else if (newHeavy.x > 0 && newHeavy.y > 0)
				quarter = UPPERRIGHT;
 
			switch (quarter) {
			case UPPERLEFT:
				prevHO.push_back(vec3(newHeavy.x, newHeavy.y - 2, newHeavy.z));
				prevHO.push_back(vec3(newHeavy.x + 2, newHeavy.y, newHeavy.z));
				prevHO.push_back(vec3(newHeavy.x + 2, newHeavy.y - 2, newHeavy.z));
				break;
			case BOTTOMRIGHT:
				prevHO.push_back(vec3(newHeavy.x - 2, newHeavy.y, newHeavy.z));
				prevHO.push_back(vec3(newHeavy.x, newHeavy.y + 2, newHeavy.z));
				prevHO.push_back(vec3(newHeavy.x - 2, newHeavy.y + 2, newHeavy.z));
				break;
			case BOTTOMLEFT:
				prevHO.push_back(vec3(newHeavy.x + 2, newHeavy.y, newHeavy.z));
				prevHO.push_back(vec3(newHeavy.x + 2, newHeavy.y + 2, newHeavy.z));
				prevHO.push_back(vec3(newHeavy.x, newHeavy.y + 2, newHeavy.z));
				break;
			case UPPERRIGHT:
				prevHO.push_back(vec3(newHeavy.x - 2, newHeavy.y, newHeavy.z));
				prevHO.push_back(vec3(newHeavy.x, newHeavy.y - 2, newHeavy.z));
				prevHO.push_back(vec3(newHeavy.x - 2, newHeavy.y - 2, newHeavy.z));
				break;
			}
		}
 
		prevHO.push_back(newHeavy);
		mass += 0.01f;
		createSheet(prevHO, mass);
	}
 
	void setCamera() {
		if (persp) {
			camera = new PerspCamera();
			currentSphere->hasCamera = true;
			camera->currSphere = currentSphere;
			camera->wEye = currentSphere->translation;
			camera->wLookat = vec3(0, 0, radius);
			camera->wVup = vec3(0, 0, 1);
		}
		else {
			for (Object* obj : spheres)
				obj->hasCamera = false;
			camera = new OrtoCamera();
			createCamera();
		}
	}
 
	vec3 MeasureBottomPoint(RubberSheet* rubberSheet, float X, float Y) {
		Dnum2 u = Dnum2((X + 1) / 2, vec2(1, 0));
		Dnum2 v = Dnum2((Y + 1) / 2, vec2(0, 1));
 
		Dnum2 x, y, z;
 
		rubberSheet->eval(u, v, x, y, z);
		return vec3(x.f, y.f, z.f);
	}
 
	void createNewSphere() {
		float R = rand() % 256 / 256.f; float G = rand() % 256 / 256.f; float B = rand() % 256 / 256.f;
		vec3 kd(R, G, B);
		vec3 ks(4, 4, 4);
		vec3 ka(0.1f, 0.1f, 0.1f);
		
		Material* mat = new Material(kd, ks, ka, 5000);
 
		Geometry* sphere = new Sphere();
 
		Object* sphereObject = new Object(phongShader, mat, sphere);
		sphereObject->bottomPoint = MeasureBottomPoint(rubberSheet, -1 + radius, -1 + radius);
		sphereObject->PutToSurface(rubberSheet);
		sphereObject->scale = vec3(1, 1, 1) * radius;
 
		spheres.push_back(sphereObject);
		currentSphere = sphereObject;
	}
 
	void Build() {
		camera = new OrtoCamera();
		phongShader = new PhongShader();
		std::vector<vec3> HO;
		createSheet(HO);
		createNewSphere();
		createCamera();
		createLights();
	}
 
	void Render() {
		RenderState state;
		state.wEye = camera->wEye;
		state.V = camera->V();
		state.P = camera->P();
		state.lights = lights;
		sheet->Draw(state);
		for (Object* obj : spheres) {
			if(obj->alive)
				if(!obj->hasCamera)
					obj->Draw(state);
		}
	}
 
	void Animate(float tstart, float tend) {
		for (int i = 0; i < spheres.size(); ++i) {
			Object* o = spheres[i];
			o->bottomPoint = o->bottomPoint + o->velocity * (tend-tstart);
			o->PutToSurface(rubberSheet);
 
			if (o->hasCamera) {
				camera->wEye = o->translation;
				Dnum2 U((o->bottomPoint.x + 1) / 2, vec2(1, 0));
				Dnum2 V((o->bottomPoint.y + 1) / 2, vec2(0, 1));
 
				Dnum2 X, Y, Z;
				rubberSheet->eval(U, V, X, Y, Z);
 
				vec3 normal = normalize(o->MeasureNormal(X, Y, Z));
				camera->wLookat = o->velocity - dot(o->velocity, normal) * normal;
			}
 
			if (length(o->velocity) > 0) 
				o->ApplyForce(rubberSheet, tstart - tend);
 
			float epsilon = 1.f;
			float repos = 0.96f;
 
			if (o->translation.x > epsilon)
				o->bottomPoint.x = -repos;
 
			if (o->translation.y > epsilon)
				o->bottomPoint.y = -repos;
 
			if (o->translation.x < -epsilon)
				o->bottomPoint.x = repos;
 
			if (o->translation.y < -epsilon) 
				o->bottomPoint.y = repos;
 
			o->PutToSurface(rubberSheet);
 
			if (o->hasMovedAlready && rubberSheet->HitHeavyObject(vec2(o->translation.x, o->translation.y))) {
				o->alive = false;
 
				if (o->hasCamera) {
					persp = false;
					setCamera();
				}
			}
		}
 
		for (int i = 0; i < lights.size(); ++i)
			lights[i].Animate(tend);
 
		currentSphere->PutToSurface(rubberSheet);
	}
 
	Object* getCurrentSphere() {
		return currentSphere;
	}
};
 
Scene scene;
 
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}
 
void onDisplay() {
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
	scene.Render();
	glutSwapBuffers();								
}
 
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == ' ') {
		if (scene.persp) {
			scene.persp = false;
			scene.getCurrentSphere()->hasCamera = false;
		}
		else
			scene.persp = true;
		scene.setCamera();
	}
}
 
void onKeyboardUp(unsigned char key, int pX, int pY) { }
 
void onMouse(int button, int state, int pX, int pY) { 
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
 
	if (button == GLUT_LEFT_BUTTON) {
		if (state == GLUT_DOWN) {
			vec3 vel = vec3(cX + 1, cY + 1, 0);
			scene.getCurrentSphere()->velocity = vel;
			scene.getCurrentSphere()->hasMovedAlready = true;
			scene.createNewSphere();
		}
	}
 
	if (button == GLUT_RIGHT_BUTTON)
		if(state == GLUT_DOWN)
			scene.addHeavyObject(vec2(cX, cY));
}
 
void onMouseMotion(int pX, int pY) {
}
 
void onIdle() {
	static float tend = 0;
	const float dt = 0.01f;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
 
	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();

}